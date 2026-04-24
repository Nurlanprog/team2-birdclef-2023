#!/usr/bin/env python3

import os
import sys
import math
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND']        = 'tensorflow'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import keras

keras.mixed_precision.set_global_policy('mixed_bfloat16')

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, top_k_accuracy_score,
    classification_report, confusion_matrix,
    precision_recall_curve, auc,
)
import sklearn.metrics



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
log = logging.getLogger(__name__)
log.info("Mixed precision policy : mixed_bfloat16")



def parse_args():
    p = argparse.ArgumentParser(
        description="BirdCLEF EfficientNetB0 + Transformer (BF16, 4x H100)"
    )
    p.add_argument("--dataset_dir",  required=True)
    p.add_argument("--path_data",    required=True)
    p.add_argument("--output_dir",   default="./outputs")
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--batch_size",   type=int,   default=128,
                   help="Per-GPU batch size")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--steps",        type=int,   default=400)
    p.add_argument("--patience",     type=int,   default=5)
    p.add_argument("--seed",         type=int,   default=887)
    p.add_argument("--debug",        action="store_true")
    p.add_argument("--fine_tune_at", type=int,   default=80,
                   help="EfficientNetB0 tail layers to unfreeze (239 total)")
    p.add_argument("--tf_dim",       type=int,   default=256)
    p.add_argument("--tf_heads",     type=int,   default=8)
    p.add_argument("--tf_blocks",    type=int,   default=2)
    p.add_argument("--tf_ff_mult",   type=int,   default=4)
    p.add_argument("--tf_dropout",   type=float, default=0.1)
    return p.parse_args()


args = parse_args()

N_LABEL   = 264
IMG_SHAPE = (128, 256, 1)
CHANNELS  = 1
LABEL_COL = "label"

AUG_PROBA       = 0.8
FREQ_MASK_PARAM = 20
TIME_MASK_PARAM = 40
NUM_FREQ_MASKS  = 2
NUM_TIME_MASKS  = 2

LR_WARMUP_EPOCHS = 3
LR_MIN           = 1e-6

OUT_DIR   = Path(args.output_dir)
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
log.info(f"Output dir : {OUT_DIR}")
log.info(f"Plots dir  : {PLOTS_DIR}")


def savefig(name):
    path = PLOTS_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Plot saved : {path}")



strategy = tf.distribute.MirroredStrategy()
N_GPUS   = strategy.num_replicas_in_sync
log.info(f"Replicas in sync : {N_GPUS}")
log.info(f"GPU devices      : {[str(d) for d in tf.config.list_logical_devices('GPU')]}")

GLOBAL_BATCH = args.batch_size * N_GPUS
log.info(f"Global batch     : {GLOBAL_BATCH}  ({args.batch_size} per GPU)")



log.info("Loading CSV ...")
data = pd.read_csv(args.path_data)
data["path_img"] = args.dataset_dir + data["filename"]

if args.debug:
    data = data.iloc[:2000]
    log.info("DEBUG MODE: 2000 rows only")

log.info(f"Total rows   : {len(data):,}")
log.info(f"Unique labels: {data[LABEL_COL].nunique()}")

min_req = math.ceil(2 / 0.2)
counts  = data[LABEL_COL].value_counts()
rare    = counts[counts < min_req].index
common  = counts[counts >= min_req].index

rare_df   = data[data[LABEL_COL].isin(rare)]
common_df = data[data[LABEL_COL].isin(common)]

log.info(f"Rare classes (<{min_req}): {len(rare)}  Common: {len(common)}")

train_common, holdout = train_test_split(
    common_df, test_size=0.2,
    stratify=common_df[LABEL_COL], random_state=args.seed,
)
valid_df, test_df = train_test_split(
    holdout, test_size=0.5,
    stratify=holdout[LABEL_COL], random_state=args.seed,
)
train_df = pd.concat([train_common, rare_df], ignore_index=True)

log.info(
    f"Split — train: {len(train_df):,}  "
    f"valid: {len(valid_df):,}  "
    f"test: {len(test_df):,}"
)
for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
    log.info(f"  {name}: {df[LABEL_COL].nunique()} unique labels")



AUTOTUNE = tf.data.AUTOTUNE


def read_image(path_img):
    img = tf.io.decode_jpeg(tf.io.read_file(path_img), channels=CHANNELS)
    img = tf.reshape(img, IMG_SHAPE)
    return tf.cast(img, tf.float32)


def decode_label(label):
    return tf.one_hot(label, depth=N_LABEL)


def freq_mask(spec, param):
    freq_bins = tf.shape(spec)[0]
    f  = tf.random.uniform([], 0, tf.minimum(param, freq_bins), dtype=tf.int32)
    f0 = tf.random.uniform([], 0, tf.maximum(freq_bins - f, 1), dtype=tf.int32)
    mask = tf.cast(
        ~((tf.range(freq_bins) >= f0) & (tf.range(freq_bins) < f0 + f)),
        spec.dtype,
    )
    return spec * tf.reshape(mask, [-1, 1, 1])


def time_mask(spec, param):
    time_steps = tf.shape(spec)[1]
    t  = tf.random.uniform([], 0, tf.minimum(param, time_steps), dtype=tf.int32)
    t0 = tf.random.uniform([], 0, tf.maximum(time_steps - t, 1), dtype=tf.int32)
    mask = tf.cast(
        ~((tf.range(time_steps) >= t0) & (tf.range(time_steps) < t0 + t)),
        spec.dtype,
    )
    return spec * tf.reshape(mask, [1, -1, 1])


def spec_augment(img):
    for _ in range(NUM_FREQ_MASKS):
        img = freq_mask(img, FREQ_MASK_PARAM)
    for _ in range(NUM_TIME_MASKS):
        img = time_mask(img, TIME_MASK_PARAM)
    return img


def augment_image(img):
    if tf.random.uniform([]) > AUG_PROBA:
        return img
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = spec_augment(img)
    return img


def mixup_batch(imgs, labels):
    batch_size = tf.shape(imgs)[0]
    lam        = tf.random.uniform([], 0.0, 1.0)
    lam        = tf.maximum(lam, 1.0 - lam)
    indices    = tf.random.shuffle(tf.range(batch_size))
    return (
        lam * imgs   + (1.0 - lam) * tf.gather(imgs,   indices),
        lam * labels + (1.0 - lam) * tf.gather(labels, indices),
    )


def make_balanced_dataset(df, augment=True):
    present = sorted(df[LABEL_COL].unique())
    class_datasets = []
    for lbl in present:
        sub = df[df[LABEL_COL] == lbl]
        ds  = (
            tf.data.Dataset.from_tensor_slices(
                (sub["path_img"].values, sub[LABEL_COL].values)
            )
            .map(
                lambda p, l: (read_image(p), decode_label(l)),
                num_parallel_calls=AUTOTUNE,
            )
            .repeat()
        )
        if augment:
            ds = ds.map(
                lambda img, lbl: (augment_image(img), lbl),
                num_parallel_calls=AUTOTUNE,
            )
        class_datasets.append(ds)
    weights = [1.0 / len(class_datasets)] * len(class_datasets)
    return tf.data.Dataset.sample_from_datasets(class_datasets, weights=weights)


def create_training_dataset(df):
    ds = make_balanced_dataset(df, augment=True)
    ds = ds.batch(GLOBAL_BATCH, drop_remainder=True)
    ds = ds.map(mixup_batch, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def create_validation_dataset(df):
    return (
        tf.data.Dataset.from_tensor_slices(
            (df["path_img"].values, df[LABEL_COL].values)
        )
        .map(
            lambda p, l: (read_image(p), decode_label(l)),
            num_parallel_calls=AUTOTUNE,
        )
        .batch(GLOBAL_BATCH)
        .prefetch(AUTOTUNE)
    )



def plot_augmentation_samples():
    log.info("Saving plot: augmentation samples ...")
    rec = data.sample(1, random_state=args.seed).iloc[0]
    img = read_image(rec.path_img)

    fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(18, 8))
    fig.suptitle(
        f"Original (top-left) vs SpecAugment + brightness/contrast variants"
        f"\nLabel: {rec[LABEL_COL]}  |  {rec['filename']}",
        fontsize=12,
    )
    for i, ax in enumerate(axs.flat):
        if i == 0:
            ax.imshow(img.numpy().squeeze(), cmap='viridis', aspect='auto')
            ax.set_title("Original", fontsize=9)
        else:
            ax.imshow(
                augment_image(img).numpy().squeeze(),
                cmap='viridis', aspect='auto',
            )
            ax.set_title(f"Aug #{i}", fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    savefig("01_augmentation_samples.png")



def plot_batch_balance_check():
    log.info("Saving plot: batch balance check ...")
    dev_data = data.sample(min(2000, len(data)), random_state=args.seed)
    dev_ds   = create_training_dataset(dev_data)

    batch_imgs, batch_labels = next(iter(dev_ds.take(1)))
    label_ids = tf.argmax(batch_labels, axis=1).numpy()

    log.info(
        f"Batch shape: {batch_imgs.shape}  "
        f"Unique labels: {len(set(label_ids))} / {GLOBAL_BATCH}"
    )

    fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(18, 8))
    fig.suptitle(
        f"Sample training batch — {len(set(label_ids))} unique labels "
        f"in {GLOBAL_BATCH} samples (balanced sampler + MixUp)",
        fontsize=12,
    )
    for i, ax in enumerate(axs.flat):
        ax.imshow(batch_imgs[i].numpy().squeeze(), cmap='viridis', aspect='auto')
        ax.set_title(f"label: {label_ids[i]}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    savefig("02_batch_balance_check.png")



class CLSTokenPrepend(keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="cls_token",
        )
        super().build(input_shape)

    def call(self, x):
        batch_size = keras.ops.shape(x)[0]
        cls_tokens = keras.ops.tile(self.cls_token, [batch_size, 1, 1])
        return keras.ops.concatenate([cls_tokens, x], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config


def transformer_encoder_block(x, dim, num_heads, ff_mult, dropout_rate, name):
    attn_in  = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=dim // num_heads,
        dropout=dropout_rate,
        name=f"{name}_mhsa",
    )(attn_in, attn_in)
    attn_out = layers.Dropout(dropout_rate, name=f"{name}_attn_drop")(attn_out)
    x        = layers.Add(name=f"{name}_add1")([x, attn_out])

    ff_in  = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    ff_h   = layers.Dense(dim * ff_mult, activation="gelu", name=f"{name}_ff1")(ff_in)
    ff_h   = layers.Dropout(dropout_rate, name=f"{name}_ff_drop1")(ff_h)
    ff_out = layers.Dense(dim, name=f"{name}_ff2")(ff_h)
    ff_out = layers.Dropout(dropout_rate, name=f"{name}_ff_drop2")(ff_out)
    x      = layers.Add(name=f"{name}_add2")([x, ff_out])
    return x



def create_model(lr, tf_dim, tf_heads, tf_blocks, tf_ff_mult,
                 tf_dropout, fine_tune_at):
    inputs = layers.Input(shape=IMG_SHAPE, dtype=tf.float32, name="input_spec")

    x = layers.Concatenate(axis=-1, name="to_rgb")([inputs, inputs, inputs])
    x = layers.Lambda(preprocess_input, name="preprocess")(x)

    backbone = EfficientNetB0(include_top=False, weights="imagenet", pooling=None)
    backbone.trainable = True
    for layer in backbone.layers[:-fine_tune_at]:
        layer.trainable = False

    n_trainable = sum(1 for l in backbone.layers if l.trainable)
    log.info(f"EfficientNetB0 trainable: {n_trainable} / {len(backbone.layers)}")

    x        = backbone(x, training=True)
    feat_dim = x.shape[-1]

    x = layers.Reshape((-1, feat_dim), name="tokenize")(x)

    x = layers.Dense(tf_dim, name="token_proj")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="token_norm")(x)

    x = CLSTokenPrepend(dim=tf_dim, name="cls_prepend")(x)

    for i in range(tf_blocks):
        x = transformer_encoder_block(
            x, dim=tf_dim, num_heads=tf_heads,
            ff_mult=tf_ff_mult, dropout_rate=tf_dropout,
            name=f"tf_block_{i}",
        )

    cls_out = x[:, 0, :]
    cls_out = layers.LayerNormalization(epsilon=1e-6, name="cls_ln")(cls_out)
    cls_out = layers.Dropout(0.1, name="cls_drop")(cls_out)
    x       = layers.Dense(256, activation="gelu", name="head_dense")(cls_out)
    x       = layers.Dropout(0.2, name="head_drop")(x)
    x       = layers.Dense(N_LABEL, name="logits")(x)

    outputs = layers.Activation("linear", dtype="float32", name="logits_f32")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="EFF-B0-Transformer")

    total_steps  = args.steps * args.epochs
    warmup_steps = args.steps * LR_WARMUP_EPOCHS
    lr_schedule  = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=max(total_steps - warmup_steps, 1),
        alpha=LR_MIN / lr,
        warmup_target=lr,
        warmup_steps=warmup_steps,
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-4,
            beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        ),
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.05,
        ),
        metrics=["acc"],
    )
    return model



def get_callbacks(run_name):
    weights = str(OUT_DIR / f"weights_{run_name}.weights.h5")
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=weights,
            monitor="val_loss", mode="auto",
            verbose=1, save_best_only=True, save_weights_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="auto",
            patience=args.patience, verbose=1, restore_best_weights=True,
        ),
        keras.callbacks.CSVLogger(
            str(OUT_DIR / f"history_{run_name}.csv"), append=False,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(OUT_DIR / "tb" / run_name),
            histogram_freq=0, update_freq="epoch",
        ),
    ]



def plot_training_history(hist, run_name):
    log.info("Saving plot: training history ...")
    hf = pd.DataFrame(hist.history)
    hf.index = pd.RangeIndex(1, len(hf) + 1, name="epoch")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Training History — {run_name}", fontsize=13)

    hf[["loss", "val_loss"]].plot(ax=axes[0], title="Loss")
    best_epoch = hf["val_loss"].idxmin()
    axes[0].axvline(best_epoch, color='red', linestyle='--', alpha=0.6,
                    label=f"Best epoch {best_epoch}")
    axes[0].legend()
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)

    hf[["acc", "val_acc"]].plot(ax=axes[1], title="Accuracy")
    best_acc_epoch = hf["val_acc"].idxmax()
    axes[1].axvline(best_acc_epoch, color='red', linestyle='--', alpha=0.6,
                    label=f"Best epoch {best_acc_epoch}")
    axes[1].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    savefig("03_training_history.png")

    log.info(
        f"Best val_loss  : {hf['val_loss'].min():.4f} at epoch {best_epoch}"
    )
    log.info(
        f"Best val_acc   : {hf['val_acc'].max():.4f} at epoch {best_acc_epoch}"
    )



def run_training(train_df, valid_df, run_name):
    train_ds = create_training_dataset(train_df)
    valid_ds = create_validation_dataset(valid_df)

    keras.backend.clear_session()
    with strategy.scope():
        model = create_model(
            lr=args.lr,
            tf_dim=args.tf_dim,
            tf_heads=args.tf_heads,
            tf_blocks=args.tf_blocks,
            tf_ff_mult=args.tf_ff_mult,
            tf_dropout=args.tf_dropout,
            fine_tune_at=args.fine_tune_at,
        )

    model.summary(line_length=120, expand_nested=False, print_fn=log.info)
    log.info(f"steps_per_epoch  : {args.steps}")
    log.info(f"precision policy : {keras.mixed_precision.global_policy().name}")

    hist = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        validation_data=valid_ds,
        callbacks=get_callbacks(run_name),
        verbose=1,
    )

    plot_training_history(hist, run_name)

    model_path = str(OUT_DIR / f"model_{run_name}.keras")
    model.save(model_path)
    log.info(f"Full model saved : {model_path}")

    return hist, model



def evaluate(model, test_df, run_name):
    log.info("Running test set evaluation ...")
    test_ds = create_validation_dataset(test_df)

    true_oh  = np.concatenate([lbl.numpy() for _, lbl in test_ds], axis=0)
    true_int = np.argmax(true_oh, axis=1)
    logits   = model.predict(test_ds, verbose=1)
    probs    = tf.nn.softmax(logits).numpy()
    pred_int = np.argmax(probs, axis=1)

    acc  = accuracy_score(true_int, pred_int)
    prec = precision_score(true_int, pred_int, average='macro', zero_division=0)
    rec  = recall_score(true_int, pred_int, average='macro', zero_division=0)
    f1   = f1_score(true_int, pred_int, average='macro', zero_division=0)
    mAP  = average_precision_score(true_oh, probs, average='macro')
    top5 = top_k_accuracy_score(true_int, probs, k=5, labels=np.arange(N_LABEL))

    n_cls      = probs.shape[1]
    cols       = list(range(n_cls))
    padding    = pd.DataFrame([[1] * n_cls] * 5, columns=cols)
    padded_sol = pd.concat(
        [pd.DataFrame(true_oh, columns=cols), padding], ignore_index=True)
    padded_sub = pd.concat(
        [pd.DataFrame(probs, columns=cols), padding], ignore_index=True)
    cmap = sklearn.metrics.average_precision_score(
        padded_sol.values, padded_sub.values, average='macro')

    results = {
        "accuracy":        round(acc,  4),
        "top5_accuracy":   round(top5, 4),
        "macro_precision": round(prec, 4),
        "macro_recall":    round(rec,  4),
        "macro_f1":        round(f1,   4),
        "macro_mAP":       round(mAP,  4),
        "padded_cMAP":     round(cmap, 4),
    }

    log.info("=" * 45)
    log.info("  TEST SET RESULTS")
    log.info("=" * 45)
    for k, v in results.items():
        log.info(f"  {k:<22} : {v:.4f}")
    log.info("=" * 45)

    out_path = OUT_DIR / f"results_{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved : {out_path}")

    report_dict = classification_report(
        true_int, pred_int, output_dict=True, zero_division=0)
    report_df = (
        pd.DataFrame(report_dict).T
        .drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        .astype(float)
    )
    report_df.index = report_df.index.astype(int)
    report_df = report_df.sort_index()

    log.info("=== Top 10 classes by F1 ===")
    log.info(report_df.nlargest(10, 'f1-score')[
        ['precision', 'recall', 'f1-score', 'support']].to_string())
    log.info("=== Bottom 10 classes by F1 ===")
    log.info(report_df.nsmallest(10, 'f1-score')[
        ['precision', 'recall', 'f1-score', 'support']].to_string())

    log.info("Saving plot: metrics vs paper ...")
    metric_names  = ['Accuracy', 'Top-5 Acc', 'Precision',
                     'Recall', 'F1', 'mAP', 'cMAP']
    metric_values = [acc, top5, prec, rec, f1, mAP, cmap]
    paper_values  = [0.8403, None, 0.8342, 0.7738, 0.7924, None, None]

    x     = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width / 2, metric_values, width,
                   label='Our model (EFF-B0+Transformer)', color='steelblue')
    bars2 = ax.bar(
        x + width / 2,
        [p if p is not None else 0 for p in paper_values],
        width,
        label='Paper baseline (EFF-B7+GRU)',
        color='coral',
        alpha=[1 if p else 0 for p in paper_values],
    )
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel('Score')
    ax.set_title(f'Our Model vs Paper Baseline — {run_name}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='steelblue')
    plt.tight_layout()
    savefig("04_metrics_vs_paper.png")

    log.info("Saving plot: F1 distribution ...")
    f1_scores = report_df['f1-score'].values
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Per-Class F1 Score Distribution', fontsize=13)

    axes[0].hist(f1_scores, bins=30, color='steelblue', edgecolor='white')
    axes[0].axvline(f1_scores.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean  = {f1_scores.mean():.3f}')
    axes[0].axvline(np.median(f1_scores), color='orange', linestyle='--',
                    linewidth=2, label=f'Median = {np.median(f1_scores):.3f}')
    axes[0].set_xlabel('F1 Score')
    axes[0].set_ylabel('Number of Classes')
    axes[0].set_title('Histogram')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    sorted_f1 = np.sort(f1_scores)
    cdf_y     = np.arange(1, len(sorted_f1) + 1) / len(sorted_f1)
    axes[1].plot(sorted_f1, cdf_y, color='steelblue', linewidth=2)
    axes[1].fill_between(sorted_f1, cdf_y, alpha=0.1, color='steelblue')
    for thresh in [0.5, 0.7, 0.9]:
        frac = (f1_scores >= thresh).mean()
        axes[1].axvline(thresh, color='gray', linestyle=':', alpha=0.7)
        axes[1].text(thresh + 0.01, 0.05,
                     f'{frac * 100:.0f}% >= {thresh}',
                     fontsize=8, color='gray')
    axes[1].set_xlabel('F1 Score')
    axes[1].set_ylabel('Cumulative Fraction of Classes')
    axes[1].set_title('CDF')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    savefig("05_f1_distribution.png")

    log.info("Saving plot: best and worst classes ...")
    n         = 15
    top_df    = report_df.nlargest(n, 'f1-score')
    bottom_df = report_df.nsmallest(n, 'f1-score')

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'Top {n} and Bottom {n} Classes by F1 Score', fontsize=13)

    for ax, df, title, color in [
        (axes[0], top_df,    f'Top {n}',    'seagreen'),
        (axes[1], bottom_df, f'Bottom {n}', 'tomato'),
    ]:
        bars = ax.barh(
            [f"class {int(i)}" for i in df.index],
            df['f1-score'], color=color, edgecolor='white',
        )
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('F1 Score')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for bar, (_, row) in zip(bars, df.iterrows()):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"F1={row['f1-score']:.2f}  n={int(row['support'])}",
                va='center', fontsize=8,
            )

    plt.tight_layout()
    savefig("06_best_worst_classes.png")

    log.info("Saving plot: confidence calibration ...")
    top1_conf    = probs.max(axis=1)
    correct_mask = (pred_int == true_int)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Model Confidence Analysis', fontsize=13)

    axes[0].hist(top1_conf[correct_mask], bins=40, alpha=0.7,
                 color='seagreen',
                 label=f'Correct ({correct_mask.sum():,})', density=True)
    axes[0].hist(top1_conf[~correct_mask], bins=40, alpha=0.7,
                 color='tomato',
                 label=f'Wrong ({(~correct_mask).sum():,})', density=True)
    axes[0].set_xlabel('Confidence (max softmax probability)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Confidence Distribution: Correct vs Wrong')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    n_bins    = 15
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (top1_conf >= lo) & (top1_conf < hi)
        if in_bin.sum() > 0:
            bin_accs.append(correct_mask[in_bin].mean())
            bin_confs.append(top1_conf[in_bin].mean())
            bin_counts.append(in_bin.sum())

    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    axes[1].scatter(bin_confs, bin_accs,
                    s=[c / 5 for c in bin_counts],
                    color='steelblue', alpha=0.8, zorder=5)
    axes[1].plot(bin_confs, bin_accs, color='steelblue', linewidth=2,
                 label='Model calibration')
    axes[1].set_xlabel('Mean Confidence')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Reliability Diagram\n(bubble size = samples in bin)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    savefig("07_confidence_calibration.png")

    log.info("Saving plot: PR curves ...")
    classes_in_test = np.unique(true_int)

    best_classes   = [c for c in report_df.nlargest(5, 'f1-score').index
                      if c in classes_in_test]
    worst_classes  = [c for c in report_df.nsmallest(5, 'f1-score').index
                      if c in classes_in_test]
    median_classes = [
        c for c in report_df['f1-score'].sort_values()
        .iloc[len(report_df) // 2 - 2: len(report_df) // 2 + 3].index
        if c in classes_in_test
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Precision-Recall Curves', fontsize=13)
    colors = plt.cm.tab10.colors

    for ax, group, title in zip(
        axes,
        [best_classes, worst_classes, median_classes],
        ['Top 5 Classes', 'Bottom 5 Classes', 'Median 5 Classes'],
    ):
        for i, cls in enumerate(group):
            y_true_bin = (true_int == cls).astype(int)
            prec_c, rec_c, _ = precision_recall_curve(y_true_bin, probs[:, cls])
            pr_auc = auc(rec_c, prec_c)
            ax.plot(rec_c, prec_c, color=colors[i], linewidth=2,
                    label=f'class {cls}  AUC={pr_auc:.2f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    savefig("08_pr_curves.png")

    log.info("Saving plot: sample predictions ...")
    n_show       = 8
    correct_idx  = np.where(pred_int == true_int)[0]
    wrong_idx    = np.where(pred_int != true_int)[0]
    correct_samp = np.random.RandomState(args.seed).choice(
        correct_idx, size=min(n_show, len(correct_idx)), replace=False)
    wrong_samp   = np.random.RandomState(args.seed).choice(
        wrong_idx, size=min(n_show, len(wrong_idx)), replace=False)

    test_df_r = test_df.reset_index(drop=True)
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 3, 8))
    fig.suptitle(
        'Sample Predictions — Top row: Correct   Bottom row: Wrong',
        fontsize=13,
    )

    for col, (c_idx, w_idx) in enumerate(zip(correct_samp, wrong_samp)):
        for row, (sample_idx, is_correct) in enumerate(
            [(c_idx, True), (w_idx, False)]
        ):
            ax    = axes[row, col]
            path  = test_df_r.iloc[sample_idx]['path_img']
            img   = read_image(path).numpy().squeeze()
            ax.imshow(img, cmap='viridis', aspect='auto', origin='lower')
            ax.axis('off')
            conf  = probs[sample_idx, pred_int[sample_idx]]
            color = 'lime' if is_correct else 'red'
            ax.set_title(
                f"True:{true_int[sample_idx]}\n"
                f"Pred:{pred_int[sample_idx]} ({conf:.2f})",
                fontsize=7, color=color, fontweight='bold',
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

    plt.tight_layout()
    savefig("09_sample_predictions.png")

    classes_above_50 = (report_df['f1-score'] >= 0.5).sum()
    classes_above_70 = (report_df['f1-score'] >= 0.7).sum()
    classes_zero     = (report_df['f1-score'] == 0.0).sum()

    log.info("=" * 54)
    log.info("  FINAL EVALUATION SUMMARY")
    log.info("=" * 54)
    log.info(f"  Accuracy              : {acc:.4f}")
    log.info(f"  Top-5 Accuracy        : {top5:.4f}")
    log.info(f"  Macro F1              : {f1:.4f}")
    log.info(f"  Macro mAP             : {mAP:.4f}")
    log.info(f"  Padded cMAP           : {cmap:.4f}  (competition metric)")
    log.info("  " + "-" * 50)
    log.info(f"  Classes with F1 >= 0.5: {classes_above_50} / {N_LABEL}")
    log.info(f"  Classes with F1 >= 0.7: {classes_above_70} / {N_LABEL}")
    log.info(f"  Classes with F1 = 0.0 : {classes_zero} / {N_LABEL}")
    log.info(f"  Mean F1 per class     : {report_df['f1-score'].mean():.4f}")
    log.info(f"  Median F1 per class   : {report_df['f1-score'].median():.4f}")
    log.info("=" * 54)

    return results



if __name__ == "__main__":
    run_name = (
        f"EFF-B0-Transformer"
        f"_blocks{args.tf_blocks}"
        f"_dim{args.tf_dim}"
        f"_heads{args.tf_heads}"
        f"_bf16"
    )
    log.info(f"Run name : {run_name}")
    log.info(f"Args     : {json.dumps(vars(args), indent=2)}")

    with open(OUT_DIR / f"config_{run_name}.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    plot_augmentation_samples()
    plot_batch_balance_check()

    hist, model = run_training(train_df, valid_df, run_name)

    results = evaluate(model, test_df, run_name)

    log.info(f"All plots saved to: {PLOTS_DIR}")
    log.info("Training complete.")


