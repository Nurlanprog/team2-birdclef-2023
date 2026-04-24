#!/usr/bin/env python3

import os
import sys
import math
import json
import logging
import argparse
from pathlib import Path

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, top_k_accuracy_score,
    classification_report, precision_recall_curve, auc,
)
import sklearn.metrics



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)



def parse_args():
    p = argparse.ArgumentParser(description="BirdCLEF visualization script")
    p.add_argument("--output_dir",  required=True,
                   help="Directory where model/weights/history were saved")
    p.add_argument("--path_data",   required=True,
                   help="Path to img_stats.csv")
    p.add_argument("--dataset_dir", required=True,
                   help="Directory containing per-species JPEG subdirs")
    p.add_argument("--model_path",  required=True,
                   help="Path to saved .keras model file")
    p.add_argument("--seed",        type=int, default=887)
    p.add_argument("--batch_size",  type=int, default=512,
                   help="Inference batch size (larger is fine, no gradients)")
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

OUT_DIR   = Path(args.output_dir)
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
log.info(f"Output dir : {OUT_DIR}")
log.info(f"Plots dir  : {PLOTS_DIR}")


def savefig(name):
    path = PLOTS_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved: {path}")



log.info("Loading data and recreating train/valid/test splits ...")
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


def create_validation_dataset(df, batch_size):
    return (
        tf.data.Dataset.from_tensor_slices(
            (df["path_img"].values, df[LABEL_COL].values)
        )
        .map(lambda p, l: (read_image(p), decode_label(l)),
             num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )



def plot_training_history():
    log.info("Plotting: training history ...")

    history_files = sorted(OUT_DIR.glob("history_*.csv"))
    if not history_files:
        log.warning("No history_*.csv found in output_dir — skipping.")
        return

    history_path = history_files[0]
    log.info(f"Loading: {history_path}")
    hf = pd.read_csv(history_path)
    hf.index = pd.RangeIndex(1, len(hf) + 1, name="epoch")

    best_loss_epoch = hf["val_loss"].idxmin()
    best_acc_epoch  = hf["val_acc"].idxmax()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Training History — {history_path.stem}", fontsize=13)

    axes[0].plot(hf.index, hf["loss"],     label="Train loss",
                 linewidth=2, color='steelblue')
    axes[0].plot(hf.index, hf["val_loss"], label="Val loss",
                 linewidth=2, color='coral', linestyle='--')
    axes[0].axvline(best_loss_epoch, color='red', linestyle=':', alpha=0.8,
                    label=f"Best epoch {best_loss_epoch} "
                          f"(val_loss={hf['val_loss'].min():.4f})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(hf.index, hf["acc"],     label="Train acc",
                 linewidth=2, color='steelblue')
    axes[1].plot(hf.index, hf["val_acc"], label="Val acc",
                 linewidth=2, color='coral', linestyle='--')
    axes[1].axvline(best_acc_epoch, color='red', linestyle=':', alpha=0.8,
                    label=f"Best epoch {best_acc_epoch} "
                          f"(val_acc={hf['val_acc'].max():.4f})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    savefig("00_training_history.png")

    log.info(
        f"Best val_loss : {hf['val_loss'].min():.4f} at epoch {best_loss_epoch}"
    )
    log.info(
        f"Best val_acc  : {hf['val_acc'].max():.4f} at epoch {best_acc_epoch}"
    )



def plot_augmentation_samples():
    log.info("Plotting: augmentation samples ...")
    rec = data.sample(1, random_state=args.seed).iloc[0]
    img = read_image(rec.path_img)

    fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(18, 8))
    fig.suptitle(
        f"Original (top-left) vs SpecAugment + brightness/contrast variants\n"
        f"Label: {rec[LABEL_COL]}  |  {rec['filename']}",
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



from keras.applications.efficientnet import preprocess_input as _preprocess_input

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


log.info(f"Loading model from : {args.model_path}")
model = keras.models.load_model(
    args.model_path,
    custom_objects={
        "preprocess_input": _preprocess_input,
        "CLSTokenPrepend":  CLSTokenPrepend,
    },
)
test_ds = create_validation_dataset(test_df, args.batch_size)

log.info("Running inference on test set ...")
true_oh  = np.concatenate([lbl.numpy() for _, lbl in test_ds], axis=0)
true_int = np.argmax(true_oh, axis=1)
logits   = model.predict(test_ds, verbose=1)
probs    = tf.nn.softmax(logits).numpy()
pred_int = np.argmax(probs, axis=1)
log.info(f"Inference complete. Test samples: {len(true_int)}")



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

log.info("=" * 45)
log.info("  TEST SET RESULTS")
log.info("=" * 45)
for name, val in [
    ("accuracy",        acc),
    ("top5_accuracy",   top5),
    ("macro_precision", prec),
    ("macro_recall",    rec),
    ("macro_f1",        f1),
    ("macro_mAP",       mAP),
    ("padded_cMAP",     cmap),
]:
    log.info(f"  {name:<22} : {val:.4f}")
log.info("=" * 45)

report_dict = classification_report(
    true_int, pred_int, output_dict=True, zero_division=0)
report_df = (
    pd.DataFrame(report_dict).T
    .drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    .astype(float)
)
report_df.index = report_df.index.astype(int)
report_df = report_df.sort_index()



plot_training_history()
plot_augmentation_samples()


log.info("Plotting: metrics vs paper ...")

metric_names  = ['Accuracy', 'Top-5 Acc', 'Precision',
                 'Recall', 'F1', 'mAP', 'cMAP']
metric_values = [acc, top5, prec, rec, f1, mAP, cmap]
paper_values  = [0.8403, None, 0.8342, 0.7738, 0.7924, None, None]

x     = np.arange(len(metric_names))
width = 0.35
fig, ax = plt.subplots(figsize=(13, 6))

bars1 = ax.bar(x - width / 2, metric_values, width,
               label='Our model (EFF-B0 + Transformer, BF16)',
               color='steelblue')

for xi, pv in zip(x + width / 2, paper_values):
    if pv is not None:
        ax.bar(xi, pv, width, color='coral', alpha=0.85, label='_nolegend_')

ax.bar([], [], color='coral', alpha=0.85,
       label='Paper baseline (EFF-B7 + GRU)')

ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylabel('Score')
ax.set_title('Our Model vs Paper Baseline', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f'{bar.get_height():.3f}',
        ha='center', va='bottom', fontsize=9,
        fontweight='bold', color='steelblue',
    )

plt.tight_layout()
savefig("02_metrics_vs_paper.png")


log.info("Plotting: F1 distribution ...")
f1_scores = report_df['f1-score'].values

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Per-Class F1 Score Distribution', fontsize=13)

axes[0].hist(f1_scores, bins=30, color='steelblue', edgecolor='white')
axes[0].axvline(f1_scores.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean  = {f1_scores.mean():.3f}')
axes[0].axvline(np.median(f1_scores), color='orange', linestyle='--',
                linewidth=2,
                label=f'Median = {np.median(f1_scores):.3f}')
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
savefig("03_f1_distribution.png")


log.info("Plotting: best and worst classes ...")
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
savefig("04_best_worst_classes.png")


log.info("Plotting: confidence calibration ...")
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
axes[0].set_title('Confidence: Correct vs Wrong Predictions')
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
savefig("05_confidence_calibration.png")


log.info("Plotting: PR curves ...")
classes_in_test = np.unique(true_int)

best_classes   = [c for c in report_df.nlargest(5,  'f1-score').index
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
        y_bin            = (true_int == cls).astype(int)
        prec_c, rec_c, _ = precision_recall_curve(y_bin, probs[:, cls])
        pr_auc           = auc(rec_c, prec_c)
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
savefig("06_pr_curves.png")


log.info("Plotting: sample predictions ...")
n_show       = 8
correct_idx  = np.where(pred_int == true_int)[0]
wrong_idx    = np.where(pred_int != true_int)[0]
rng          = np.random.RandomState(args.seed)
correct_samp = rng.choice(correct_idx,
                           size=min(n_show, len(correct_idx)), replace=False)
wrong_samp   = rng.choice(wrong_idx,
                           size=min(n_show, len(wrong_idx)),   replace=False)

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
            f"True: {true_int[sample_idx]}\n"
            f"Pred: {pred_int[sample_idx]} ({conf:.2f})",
            fontsize=7, color=color, fontweight='bold',
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

plt.tight_layout()
savefig("07_sample_predictions.png")



classes_above_50 = (report_df['f1-score'] >= 0.5).sum()
classes_above_70 = (report_df['f1-score'] >= 0.7).sum()
classes_zero     = (report_df['f1-score'] == 0.0).sum()

log.info("=" * 54)
log.info("  FINAL SUMMARY")
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
log.info(f"All 8 plots saved to: {PLOTS_DIR}")
