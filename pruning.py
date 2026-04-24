#!/usr/bin/env python3

import os
import sys
import math
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND']        = 'tensorflow'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
keras.mixed_precision.set_global_policy('mixed_bfloat16')

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score
import sklearn.metrics



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"pruning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
log = logging.getLogger(__name__)



def parse_args():
    p = argparse.ArgumentParser(
        description="Lottery Ticket Hypothesis pruning for BirdCLEF model"
    )
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--path_data",      required=True)
    p.add_argument("--dataset_dir",    required=True)
    p.add_argument("--model_path",     required=True,
                   help="Path to saved .keras model file")
    p.add_argument("--weights_path",   required=True,
                   help="Path to saved .weights.h5 (original values for reset)")
    p.add_argument("--seed",           type=int,   default=887)
    p.add_argument("--batch_size",     type=int,   default=128,
                   help="Per-GPU batch size")
    p.add_argument("--retrain_epochs", type=int,   default=20)
    p.add_argument("--retrain_steps",  type=int,   default=400)
    p.add_argument("--lr",             type=float, default=5e-5,
                   help="LR for retraining (lower than original training)")
    p.add_argument("--patience",       type=int,   default=5)
    p.add_argument(
        "--prune_rates",
        nargs="+", type=float,
        default=[0.20, 0.40, 0.60, 0.80],
        help="Cumulative sparsity targets per round",
    )
    p.add_argument(
        "--prune_scope",
        choices=["head_only", "full"],
        default="head_only",
        help=(
            "head_only: prune Transformer + classification head only. "
            "full: prune all Dense/Conv including backbone."
        ),
    )
    return p.parse_args()


args = parse_args()

N_LABEL          = 264
IMG_SHAPE        = (128, 256, 1)
CHANNELS         = 1
LABEL_COL        = "label"
AUTOTUNE         = tf.data.AUTOTUNE
LR_MIN           = 1e-7
LR_WARMUP_EPOCHS = 2

OUT_DIR     = Path(args.output_dir)
PRUNING_DIR = OUT_DIR / "pruning"
PRUNING_DIR.mkdir(parents=True, exist_ok=True)
log.info(f"Pruning output dir: {PRUNING_DIR}")



strategy     = tf.distribute.MirroredStrategy()
N_GPUS       = strategy.num_replicas_in_sync
GLOBAL_BATCH = args.batch_size * N_GPUS
log.info(f"Replicas: {N_GPUS}  |  Global batch: {GLOBAL_BATCH}")



log.info("Loading data ...")
data = pd.read_csv(args.path_data)
data["path_img"] = args.dataset_dir + data["filename"]

min_req   = math.ceil(2 / 0.2)
counts    = data[LABEL_COL].value_counts()
rare      = counts[counts < min_req].index
common    = counts[counts >= min_req].index
rare_df   = data[data[LABEL_COL].isin(rare)]
common_df = data[data[LABEL_COL].isin(common)]

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
    f"train={len(train_df):,}  "
    f"valid={len(valid_df):,}  "
    f"test={len(test_df):,}"
)



def read_image(path_img):
    img = tf.io.decode_jpeg(tf.io.read_file(path_img), channels=CHANNELS)
    img = tf.reshape(img, IMG_SHAPE)
    return tf.cast(img, tf.float32)


def decode_label(label):
    return tf.one_hot(label, depth=N_LABEL)


def freq_mask(spec, param=20):
    freq_bins = tf.shape(spec)[0]
    f  = tf.random.uniform([], 0, tf.minimum(param, freq_bins), dtype=tf.int32)
    f0 = tf.random.uniform([], 0, tf.maximum(freq_bins - f, 1), dtype=tf.int32)
    mask = tf.cast(
        ~((tf.range(freq_bins) >= f0) & (tf.range(freq_bins) < f0 + f)),
        spec.dtype,
    )
    return spec * tf.reshape(mask, [-1, 1, 1])


def time_mask(spec, param=40):
    time_steps = tf.shape(spec)[1]
    t  = tf.random.uniform([], 0, tf.minimum(param, time_steps), dtype=tf.int32)
    t0 = tf.random.uniform([], 0, tf.maximum(time_steps - t, 1), dtype=tf.int32)
    mask = tf.cast(
        ~((tf.range(time_steps) >= t0) & (tf.range(time_steps) < t0 + t)),
        spec.dtype,
    )
    return spec * tf.reshape(mask, [1, -1, 1])


def augment_image(img):
    if tf.random.uniform([]) > 0.8:
        return img
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    for _ in range(2):
        img = freq_mask(img)
    for _ in range(2):
        img = time_mask(img)
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
    present        = sorted(df[LABEL_COL].unique())
    class_datasets = []
    for lbl in present:
        sub = df[df[LABEL_COL] == lbl]
        ds  = (
            tf.data.Dataset.from_tensor_slices(
                (sub["path_img"].values, sub[LABEL_COL].values)
            )
            .map(lambda p, l: (read_image(p), decode_label(l)),
                 num_parallel_calls=AUTOTUNE)
            .repeat()
        )
        if augment:
            ds = ds.map(lambda img, lbl: (augment_image(img), lbl),
                        num_parallel_calls=AUTOTUNE)
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
        .map(lambda p, l: (read_image(p), decode_label(l)),
             num_parallel_calls=AUTOTUNE)
        .batch(GLOBAL_BATCH)
        .prefetch(AUTOTUNE)
    )



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


CUSTOM_OBJECTS = {
    "preprocess_input": preprocess_input,
    "CLSTokenPrepend":  CLSTokenPrepend,
}



def get_prunable_layers(model, scope="head_only"):
    prunable      = []
    backbone_name = "efficientnetb0"
    for layer in model.layers:
        if isinstance(layer, (
            keras.layers.BatchNormalization,
            keras.layers.LayerNormalization,
            keras.layers.Embedding,
        )):
            continue
        if scope == "head_only" and layer.name == backbone_name:
            continue
        if isinstance(layer, keras.Model):
            if scope == "full":
                for sublayer in layer.layers:
                    if hasattr(sublayer, "kernel") and sublayer.trainable:
                        prunable.append(sublayer)
            continue
        if hasattr(layer, "kernel") and layer.trainable:
            prunable.append(layer)
    log.info(
        f"Prunable layers ({scope}): {len(prunable)} — "
        + ", ".join(l.name for l in prunable[:8])
        + ("..." if len(prunable) > 8 else "")
    )
    return prunable


def compute_global_threshold(prunable_layers, target_sparsity):
    all_w     = np.concatenate([
        np.abs(l.kernel.numpy().flatten()) for l in prunable_layers
    ])
    threshold = np.percentile(all_w, target_sparsity * 100)
    total     = len(all_w)
    pruned    = (all_w <= threshold).sum()
    log.info(
        f"Target: {target_sparsity:.0%}  |  Threshold: {threshold:.6f}  |  "
        f"Actual: {pruned/total:.2%}  |  Weights: {total:,}  |  Pruned: {pruned:,}"
    )
    return threshold


def build_masks(prunable_layers, threshold):
    masks = {}
    for layer in prunable_layers:
        w               = layer.kernel.numpy()
        mask            = (np.abs(w) > threshold).astype(np.float32)
        masks[layer.name] = mask
        log.info(
            f"  {layer.name:<40} "
            f"sparsity={1-mask.mean():.2%}  shape={w.shape}"
        )
    return masks


def apply_masks(model, masks):
    for layer in model.layers:
        if layer.name in masks:
            layer.kernel.assign(layer.kernel.numpy() * masks[layer.name])
        if isinstance(layer, keras.Model):
            for sublayer in layer.layers:
                if sublayer.name in masks:
                    sublayer.kernel.assign(
                        sublayer.kernel.numpy() * masks[sublayer.name]
                    )


def reset_to_original_weights(model, weights_path):
    log.info(f"Resetting to original weights: {weights_path}")
    model.load_weights(weights_path)


def compute_sparsity_stats(model, prunable_layers):
    total    = sum(l.kernel.numpy().size for l in prunable_layers)
    nonzero  = sum((l.kernel.numpy() != 0).sum() for l in prunable_layers)
    all_p    = sum(np.prod(v.shape) for v in model.trainable_variables)
    sparsity = (1.0 - nonzero / total) if total > 0 else 0.0
    return {
        "total_params":      int(all_p),
        "prunable_params":   int(total),
        "nonzero_params":    int(nonzero),
        "pruned_params":     int(total - nonzero),
        "sparsity":          round(sparsity, 4),
        "model_size_mb":     round(all_p * 2 / 1e6, 2),
        "effective_size_mb": round(nonzero * 2 / 1e6, 2),
    }


def evaluate_model(model, test_df, label=""):
    test_ds  = create_validation_dataset(test_df)
    true_oh  = np.concatenate([lbl.numpy() for _, lbl in test_ds], axis=0)
    true_int = np.argmax(true_oh, axis=1)
    logits   = model.predict(test_ds, verbose=0)
    probs    = tf.nn.softmax(logits).numpy()
    pred_int = np.argmax(probs, axis=1)

    acc = accuracy_score(true_int, pred_int)
    mAP = average_precision_score(true_oh, probs, average='macro')

    n      = probs.shape[1]
    cols   = list(range(n))
    pad    = pd.DataFrame([[1] * n] * 5, columns=cols)
    p_sol  = pd.concat([pd.DataFrame(true_oh, columns=cols), pad],
                       ignore_index=True)
    p_sub  = pd.concat([pd.DataFrame(probs,   columns=cols), pad],
                       ignore_index=True)
    cmap   = sklearn.metrics.average_precision_score(
        p_sol.values, p_sub.values, average='macro')

    log.info(
        f"[{label}]  acc={acc:.4f}  mAP={mAP:.4f}  padded_cMAP={cmap:.4f}"
    )
    return {"accuracy": round(acc,4), "macro_mAP": round(mAP,4),
            "padded_cMAP": round(cmap,4)}



class MaskEnforcementCallback(keras.callbacks.Callback):
    def __init__(self, masks):
        super().__init__()
        self.masks = masks

    def on_train_batch_end(self, batch, logs=None):
        apply_masks(self.model, self.masks)



def plot_pruning_summary(all_results):
    log.info("Saving pruning summary plot ...")
    labels   = [r["label"]             for r in all_results]
    sparsity = [r["sparsity"] * 100    for r in all_results]
    acc      = [r["accuracy"]          for r in all_results]
    cmap     = [r["padded_cMAP"]       for r in all_results]
    size_mb  = [r["effective_size_mb"] for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Lottery Ticket Hypothesis — Pruning Results\n"
        "EfficientNetB0 + Transformer | BirdCLEF 2023",
        fontsize=13, fontweight='bold',
    )

    axes[0].plot(sparsity, acc, 'o-', linewidth=2, markersize=8, color='steelblue')
    for s, a in zip(sparsity, acc):
        axes[0].annotate(f'{a:.3f}', (s, a),
                         textcoords="offset points", xytext=(0,10),
                         ha='center', fontsize=8)
    axes[0].axhline(acc[0], color='red', linestyle='--', alpha=0.6,
                    label=f'Baseline = {acc[0]:.3f}')
    axes[0].set_xlabel('Sparsity (%)')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Accuracy vs Sparsity')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(-5, 90)

    axes[1].plot(sparsity, cmap, 'o-', linewidth=2, markersize=8, color='seagreen')
    for s, c in zip(sparsity, cmap):
        axes[1].annotate(f'{c:.3f}', (s, c),
                         textcoords="offset points", xytext=(0,10),
                         ha='center', fontsize=8)
    axes[1].axhline(cmap[0], color='red', linestyle='--', alpha=0.6,
                    label=f'Baseline = {cmap[0]:.3f}')
    axes[1].set_xlabel('Sparsity (%)')
    axes[1].set_ylabel('Padded cMAP')
    axes[1].set_title('Padded cMAP vs Sparsity\n(competition metric)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(-5, 90)

    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(all_results)))
    bars   = axes[2].bar(labels, size_mb, color=colors, edgecolor='white')
    for bar, sz in zip(bars, size_mb):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f'{sz:.1f} MB', ha='center', fontsize=9,
        )
    axes[2].set_ylabel('Effective Model Size (MB, BF16)')
    axes[2].set_title('Model Size vs Pruning Round')
    axes[2].grid(axis='y', alpha=0.3)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=20, ha='right')

    plt.tight_layout()
    path = PRUNING_DIR / "pruning_summary.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved: {path}")



def main():
    log.info("=" * 62)
    log.info("  LOTTERY TICKET HYPOTHESIS PRUNING")
    log.info("  Frankle & Carbin, ICLR 2019  arXiv:1803.03635")
    log.info("=" * 62)
    log.info(f"Prune rates    : {args.prune_rates}")
    log.info(f"Prune scope    : {args.prune_scope}")
    log.info(f"Retrain epochs : {args.retrain_epochs}")
    log.info(f"Retrain LR     : {args.lr}")

    all_results      = []
    cumulative_masks = {}

    with strategy.scope():

        keras.backend.clear_session()
        log.info(f"Loading model: {args.model_path}")
        model = keras.models.load_model(
            args.model_path,
            custom_objects=CUSTOM_OBJECTS,
        )

        prunable_layers = get_prunable_layers(model, scope=args.prune_scope)

        log.info("\n" + "=" * 62)
        log.info("ROUND 0 — Baseline (no pruning)")
        log.info("=" * 62)

        stats   = compute_sparsity_stats(model, prunable_layers)
        metrics = evaluate_model(model, test_df, label="Baseline")

        baseline_result = {
            "round": 0, "label": "Baseline", "sparsity": 0.0,
            **stats, **metrics,
        }
        all_results.append(baseline_result)

        d = PRUNING_DIR / "round_0_baseline"
        d.mkdir(exist_ok=True)
        with open(d / "results.json", "w") as f:
            json.dump(baseline_result, f, indent=2)

        log.info(f"Baseline — {json.dumps(metrics)}")
        log.info(f"Stats    — {json.dumps(stats)}")

        for round_idx, target_sparsity in enumerate(args.prune_rates, start=1):
            log.info("\n" + "=" * 62)
            log.info(f"ROUND {round_idx} — Target sparsity: {target_sparsity:.0%}")
            log.info("=" * 62)

            round_label = f"Sparse {int(target_sparsity*100)}%"
            run_dir = (
                PRUNING_DIR
                / f"round_{round_idx}_sparsity{int(target_sparsity*100)}"
            )
            run_dir.mkdir(exist_ok=True)

            threshold = compute_global_threshold(prunable_layers, target_sparsity)

            new_masks = build_masks(prunable_layers, threshold)

            for name, mask in new_masks.items():
                cumulative_masks[name] = (
                    cumulative_masks[name] * mask
                    if name in cumulative_masks
                    else mask
                )

            reset_to_original_weights(model, args.weights_path)
            apply_masks(model, cumulative_masks)

            stats_pre = compute_sparsity_stats(model, prunable_layers)
            log.info(
                f"After reset+mask — sparsity: {stats_pre['sparsity']:.2%}  "
                f"nonzero: {stats_pre['nonzero_params']:,}"
            )

            total_steps  = args.retrain_steps * args.retrain_epochs
            warmup_steps = args.retrain_steps * LR_WARMUP_EPOCHS
            lr_schedule  = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=args.lr,
                decay_steps=max(total_steps - warmup_steps, 1),
                alpha=LR_MIN / args.lr,
                warmup_target=args.lr,
                warmup_steps=warmup_steps,
            )
            model.compile(
                optimizer=keras.optimizers.AdamW(
                    learning_rate=lr_schedule,
                    weight_decay=1e-4,
                    beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                ),
                loss=keras.losses.CategoricalCrossentropy(
                    from_logits=True, label_smoothing=0.05),
                metrics=["acc"],
            )

            log.info(
                f"Retraining for {args.retrain_epochs} epochs "
                f"at lr={args.lr} ..."
            )
            train_ds = create_training_dataset(train_df)
            valid_ds = create_validation_dataset(valid_df)

            model.fit(
                train_ds,
                epochs=args.retrain_epochs,
                steps_per_epoch=args.retrain_steps,
                validation_data=valid_ds,
                callbacks=[
                    keras.callbacks.ModelCheckpoint(
                        filepath=str(run_dir / "weights.weights.h5"),
                        monitor="val_loss", mode="auto",
                        save_best_only=True, save_weights_only=True, verbose=1,
                    ),
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=args.patience,
                        restore_best_weights=True, verbose=1,
                    ),
                    keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
                    MaskEnforcementCallback(cumulative_masks),
                ],
                verbose=1,
            )

            apply_masks(model, cumulative_masks)

            stats   = compute_sparsity_stats(model, prunable_layers)
            metrics = evaluate_model(model, test_df, label=round_label)

            round_result = {
                "round": round_idx, "label": round_label,
                "sparsity": stats["sparsity"], **stats, **metrics,
            }
            all_results.append(round_result)

            with open(run_dir / "results.json", "w") as f:
                json.dump(round_result, f, indent=2)

            log.info(f"Round {round_idx} metrics : {json.dumps(metrics)}")
            log.info(f"Round {round_idx} stats   : {json.dumps(stats)}")

    results_path = PRUNING_DIR / "pruning_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nAll results saved: {results_path}")

    log.info("\n" + "=" * 82)
    log.info("  PRUNING RESULTS SUMMARY")
    log.info("=" * 82)
    log.info(
        f"{'Round':<8} {'Label':<15} {'Sparsity':>10} "
        f"{'Accuracy':>10} {'mAP':>8} {'cMAP':>8} "
        f"{'MB':>8} {'Nonzero':>12}"
    )
    log.info("-" * 82)
    for r in all_results:
        log.info(
            f"{r['round']:<8} {r['label']:<15} "
            f"{r['sparsity']*100:>9.1f}% "
            f"{r['accuracy']:>10.4f} "
            f"{r['macro_mAP']:>8.4f} "
            f"{r['padded_cMAP']:>8.4f} "
            f"{r['effective_size_mb']:>7.1f} "
            f"{r['nonzero_params']:>12,}"
        )
    log.info("=" * 82)

    plot_pruning_summary(all_results)
    log.info(f"\nPruning complete. All outputs: {PRUNING_DIR}")


if __name__ == "__main__":
    main()


