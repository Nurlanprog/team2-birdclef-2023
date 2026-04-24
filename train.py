#!/usr/bin/env python3

import os
import sys
import argparse
import random
import math
import json
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as T
import timm

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, log_loss, average_precision_score,
    classification_report,
)

from torchmetrics import Accuracy, AUROC

warnings.filterwarnings("ignore")


class CFG:
    data_dir = "/shared/home/hari.rai/Nurlan/experiment/birdclef-2023"
    train_audio_dir = os.path.join(data_dir, "train_audio")
    metadata_path = os.path.join(data_dir, "train_metadata.csv")
    output_dir = "/shared/home/hari.rai/Nurlan/experiment/birdclef-2023/outputs"

    sample_rate = 32000
    duration = 5
    audio_len = sample_rate * duration

    n_mels = 128
    n_fft = 2048
    hop_length = 2048
    f_min = 500
    f_max = 12500

    model_name = "tf_efficientnet_b7_ns"
    freeze_layers = 612
    gru_hidden = 256
    gru_layers = 1
    dropout = 0.3

    epochs = 70
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-5
    label_smoothing = 0.05
    warmup_epochs = 3
    num_workers = 8

    mixup_alpha = 0.4
    mixup_prob = 0.5
    time_mask_param = 25
    freq_mask_param = 10

    seed = 42
    debug = False
    gpus = 4


class BirdCLEFDataset(Dataset):
    def __init__(self, df, class_to_idx, cfg, augment=False):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.cfg = cfg
        self.augment = augment
        self.num_classes = len(class_to_idx)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2.0,
        )
        self.db_transform = T.AmplitudeToDB(stype="power", top_db=80)
        self.time_mask = T.TimeMasking(time_mask_param=cfg.time_mask_param)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=cfg.freq_mask_param)

    def __len__(self):
        return len(self.df)

    def _load_audio(self, filepath):
        try:
            audio, sr = torchaudio.load(filepath)
        except Exception:
            try:
                audio_np, sr = librosa.load(filepath, sr=self.cfg.sample_rate, mono=True)
                audio = torch.from_numpy(audio_np).unsqueeze(0)
                sr = self.cfg.sample_rate
            except Exception:
                return torch.zeros(1, self.cfg.audio_len)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != self.cfg.sample_rate:
            audio = T.Resample(sr, self.cfg.sample_rate)(audio)

        target_len = self.cfg.audio_len
        if audio.shape[1] > target_len:
            start = random.randint(0, audio.shape[1] - target_len) if self.augment else 0
            audio = audio[:, start:start + target_len]
        elif audio.shape[1] < target_len:
            repeats = math.ceil(target_len / audio.shape[1])
            audio = audio.repeat(1, repeats)[:, :target_len]
        return audio

    def _to_melspec(self, audio):
        mel = self.mel_transform(audio)
        mel = self.db_transform(mel)
        mel_min, mel_max = mel.min(), mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-8)
        mel = mel.repeat(3, 1, 1)
        if self.augment:
            mel = self.time_mask(mel)
            mel = self.time_mask(mel)
            mel = self.freq_mask(mel)
        return mel

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.cfg.train_audio_dir, row["primary_label"], row["filename"])
        label = self.class_to_idx[row["primary_label"]]
        audio = self._load_audio(filepath)
        mel = self._to_melspec(audio)
        return mel, label


class BirdCLEFDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.class_to_idx = None
        self.idx_to_class = None
        self.num_classes = None

    def setup(self, stage=None):
        df = pd.read_csv(self.cfg.metadata_path)
        print(f"[Data] Total samples: {len(df)} | Species: {df['primary_label'].nunique()}")

        class_names = sorted(df["primary_label"].unique().tolist())
        self.num_classes = len(class_names)
        self.class_to_idx = {n: i for i, n in enumerate(class_names)}
        self.idx_to_class = {i: n for n, i in self.class_to_idx.items()}

        if self.cfg.debug:
            df = df.groupby("primary_label").head(5).reset_index(drop=True)
            print(f"[Debug] Subsampled to {len(df)}")

        counts = df["primary_label"].value_counts()
        singleton_mask = df["primary_label"].isin(counts[counts < 2].index)
        singleton_df   = df[singleton_mask]
        splittable_df  = df[~singleton_mask]

        n_singletons = len(singleton_df)
        if n_singletons:
            print(f"[Data] Singleton classes (sent straight to train): {n_singletons} samples")

        train_df, temp_df = train_test_split(
            splittable_df,
            test_size=0.2,
            random_state=self.cfg.seed,
            stratify=splittable_df["primary_label"],
        )

        train_df = pd.concat([train_df, singleton_df], ignore_index=True)

        counts_temp        = temp_df["primary_label"].value_counts()
        singleton_temp_mask = temp_df["primary_label"].isin(counts_temp[counts_temp < 2].index)
        singleton_temp_df  = temp_df[singleton_temp_mask]
        splittable_temp_df = temp_df[~singleton_temp_mask]

        val_df, test_df = train_test_split(
            splittable_temp_df,
            test_size=0.5,
            random_state=self.cfg.seed,
            stratify=splittable_temp_df["primary_label"],
        )

        val_df = pd.concat([val_df, singleton_temp_df], ignore_index=True)

        median_count = int(train_df["primary_label"].value_counts().median())
        target = max(median_count, 20)
        upsampled = []
        for label, group in train_df.groupby("primary_label"):
            if len(group) < target:
                upsampled.append(group.sample(target, replace=True, random_state=self.cfg.seed))
            else:
                upsampled.append(group)
        train_df = pd.concat(upsampled, ignore_index=True)

        print(f"[Data] Train: {len(train_df)} (upsampled) | Val: {len(val_df)} | Test: {len(test_df)}")

        self.train_ds = BirdCLEFDataset(train_df, self.class_to_idx, self.cfg, augment=True)
        self.val_ds   = BirdCLEFDataset(val_df,   self.class_to_idx, self.cfg, augment=False)
        self.test_ds  = BirdCLEFDataset(test_df,  self.class_to_idx, self.cfg, augment=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True, drop_last=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.cfg.batch_size * 2, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.cfg.batch_size * 2, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
        )


class EfficientNetGRU(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.model_name, pretrained=True, features_only=False,
            num_classes=0, global_pool="",
        )

        all_params = list(self.backbone.named_parameters())
        for i, (_, p) in enumerate(all_params):
            if i < cfg.freeze_layers:
                p.requires_grad = False
        trainable = sum(p.requires_grad for _, p in all_params)
        print(f"[Model] Backbone params: {len(all_params)} total | "
              f"Frozen: {len(all_params)-trainable} | Trainable: {trainable}")

        with torch.no_grad():
            dummy = torch.randn(1, 3, cfg.n_mels, 80)
            feat = self.backbone(dummy)
        self.feat_c = feat.shape[1]
        self.feat_h = feat.shape[2] if len(feat.shape) == 4 else 1

        gru_input = self.feat_c * self.feat_h
        self.gru = nn.GRU(
            input_size=gru_input, hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers, batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.gru_hidden * 2, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        B, C, H, W = feat.shape
        feat = feat.reshape(B, C * H, W)
        feat = feat.permute(0, 2, 1)
        gru_out, _ = self.gru(feat)
        out = gru_out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


class BirdCLEFModule(L.LightningModule):
    def __init__(self, num_classes: int, cfg, class_to_idx: dict, idx_to_class: dict):
        super().__init__()
        self.save_hyperparameters(ignore=["class_to_idx", "idx_to_class"])
        self.cfg = cfg
        self.num_classes = num_classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class

        self.model = EfficientNetGRU(num_classes, cfg)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)

        self.val_preds: List[torch.Tensor] = []
        self.val_probs: List[torch.Tensor] = []
        self.val_targets: List[torch.Tensor] = []
        self.test_preds: List[torch.Tensor] = []
        self.test_probs: List[torch.Tensor] = []
        self.test_targets: List[torch.Tensor] = []

    @staticmethod
    def mixup(x, y, alpha=0.4):
        lam = max(np.random.beta(alpha, alpha), 1 - np.random.beta(alpha, alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        use_mixup = random.random() < self.cfg.mixup_prob
        if use_mixup:
            x, ya, yb, lam = self.mixup(x, y, self.cfg.mixup_alpha)
            logits = self(x)
            loss = lam * self.criterion(logits, ya) + (1 - lam) * self.criterion(logits, yb)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)

        preds = logits.argmax(dim=1)
        self.train_acc(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        self.val_acc(preds, y)
        self.val_preds.append(preds.cpu())
        self.val_probs.append(probs.cpu())
        self.val_targets.append(y.cpu())

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc",  self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return
        preds   = torch.cat(self.val_preds).numpy()
        probs   = torch.cat(self.val_probs).numpy()
        targets = torch.cat(self.val_targets).numpy()
        self.val_preds.clear(); self.val_probs.clear(); self.val_targets.clear()

        m = self._sklearn_metrics(targets, preds, probs)
        for k, v in m.items():
            self.log(f"val/{k}", v, prog_bar=(k in ("macro_f1", "padded_cmap")), sync_dist=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        self.test_preds.append(preds.cpu())
        self.test_probs.append(probs.cpu())
        self.test_targets.append(y.cpu())
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self):
        preds   = torch.cat(self.test_preds).numpy()
        probs   = torch.cat(self.test_probs).numpy()
        targets = torch.cat(self.test_targets).numpy()
        self.test_preds.clear(); self.test_probs.clear(); self.test_targets.clear()

        m = self._sklearn_metrics(targets, preds, probs)

        print(f"\n{'='*70}")
        print("FINAL TEST RESULTS  (replicating Table 3 from paper)")
        print(f"{'='*70}")
        print(f"  Accuracy:            {m['accuracy']*100:.2f}%")
        print(f"  Macro Avg Precision: {m['macro_precision']:.4f}")
        print(f"  Recall:              {m['macro_recall']:.4f}")
        print(f"  F1-Score:            {m['macro_f1']:.4f}")
        print(f"  Cohen's Kappa:       {m['cohen_kappa']:.4f}")
        print(f"  Log Loss:            {m['log_loss']:.4f}")
        print(f"  Padded cmAP:         {m['padded_cmap']:.4f}")
        print(f"{'='*70}")
        print("Paper reference  (EfficientNet-B7 + GRU):")
        print("  Acc 84.03% | mAP 0.8342 | Recall 0.7738 | "
              "F1 0.7924 | Kappa 0.8376 | LogLoss 3.65")
        print(f"{'='*70}\n")

        for k, v in m.items():
            self.log(f"test/{k}", v, sync_dist=False)

        self._save_results(m, targets, preds, probs)

    def _sklearn_metrics(self, targets, preds, probs) -> Dict[str, float]:
        acc   = accuracy_score(targets, preds)
        prec  = precision_score(targets, preds, average="macro", zero_division=0)
        rec   = recall_score(targets, preds, average="macro", zero_division=0)
        f1    = f1_score(targets, preds, average="macro", zero_division=0)
        kappa = cohen_kappa_score(targets, preds)

        y_oh = np.zeros((len(targets), self.num_classes))
        for i, t in enumerate(targets):
            y_oh[i, t] = 1.0
        try:
            ll = log_loss(y_oh, probs, labels=list(range(self.num_classes)))
        except Exception:
            ll = float("inf")

        pad = np.ones((5, self.num_classes))
        try:
            cmap = average_precision_score(
                np.concatenate([y_oh, pad]), np.concatenate([probs, pad]), average="macro"
            )
        except Exception:
            cmap = 0.0

        return dict(accuracy=acc, macro_precision=prec, macro_recall=rec,
                    macro_f1=f1, cohen_kappa=kappa, log_loss=ll, padded_cmap=cmap)

    def _save_results(self, metrics, targets, preds, probs):
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        results = {
            "test_metrics": {k: float(v) for k, v in metrics.items()},
            "paper_reference": dict(
                model="EfficientNet-B7 + GRU", accuracy=0.8403,
                macro_precision=0.8342, recall=0.7738, f1=0.7924,
                cohen_kappa=0.8376, log_loss=3.65,
            ),
        }
        with open(os.path.join(self.cfg.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        report = classification_report(
            targets, preds,
            target_names=[self.idx_to_class[i] for i in range(self.num_classes)],
            output_dict=True, zero_division=0,
        )
        with open(os.path.join(self.cfg.output_dir, "classification_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        with open(os.path.join(self.cfg.output_dir, "class_to_idx.json"), "w") as f:
            json.dump(self.class_to_idx, f, indent=2)

        print(f"[Save] Results, classification report, class map → {self.cfg.output_dir}")

    def configure_optimizers(self):
        backbone_params, head_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in ("gru", "fc", "dropout")):
                head_params.append(p)
            else:
                backbone_params.append(p)

        optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": self.cfg.lr * 0.1},
            {"params": head_params,     "lr": self.cfg.lr},
        ], weight_decay=self.cfg.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.epochs, eta_min=1e-7,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def plot_results(output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_dir = os.path.join(output_dir, "lightning_logs")
    versions = sorted(Path(log_dir).glob("version_*"), key=lambda p: int(p.name.split("_")[1]))
    if not versions:
        print("No logs found."); return
    metrics_file = versions[-1] / "metrics.csv"
    if not metrics_file.exists():
        print(f"No metrics.csv in {versions[-1]}"); return

    mdf = pd.read_csv(metrics_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epoch_loss = mdf.dropna(subset=["train/loss_epoch"]).groupby("epoch")["train/loss_epoch"].last()
    val_loss   = mdf.dropna(subset=["val/loss"]).groupby("epoch")["val/loss"].last()
    ax1.plot(epoch_loss.index, epoch_loss.values, "b-o", ms=3, label="Train Loss")
    ax1.plot(val_loss.index, val_loss.values, "r-o", ms=3, label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_title("Loss")

    train_acc = mdf.dropna(subset=["train/acc"]).groupby("epoch")["train/acc"].last()
    val_acc   = mdf.dropna(subset=["val/acc"]).groupby("epoch")["val/acc"].last()
    ax2.plot(train_acc.index, train_acc.values * 100, "b-o", ms=3, label="Train Acc")
    ax2.plot(val_acc.index, val_acc.values * 100, "r-o", ms=3, label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_title("Accuracy")

    fig.suptitle("EfficientNet-B7 + GRU — BirdCLEF 2023 (FP16)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir}/training_curves.png")
    plt.close()

    results_file = os.path.join(output_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            res = json.load(f)
        ours  = res["test_metrics"]
        paper = res["paper_reference"]
        keys = [("accuracy","Accuracy"), ("macro_precision","Macro Precision"),
                ("macro_recall","Recall"), ("macro_f1","F1"), ("cohen_kappa","Kappa")]
        labels = [k[1] for k in keys]
        ov = [ours.get(k[0],0) for k in keys]
        pv = [paper.get(k[0],0) for k in keys]
        x = np.arange(len(labels)); w = 0.35
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x-w/2, pv, w, label="Paper (EffNet-B7+GRU)", color="steelblue")
        ax.bar(x+w/2, ov, w, label="Ours (Replication, FP16)", color="coral")
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend()
        ax.set_ylim(0,1); ax.grid(axis="y", alpha=0.3)
        ax.set_title("Test Metrics vs Paper", fontsize=14, weight="bold")
        for b in ax.patches:
            ax.annotate(f"{b.get_height():.3f}", (b.get_x()+b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "paper_comparison.png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir}/paper_comparison.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",       action="store_true")
    parser.add_argument("--gpus",        type=int,   default=4)
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--output_dir",  type=str,   default=None)
    parser.add_argument("--data_dir",    type=str,   default=None)
    parser.add_argument("--num_workers", type=int,   default=None)
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Path to checkpoint to resume training")
    parser.add_argument("--plot_only",   action="store_true",
                        help="Skip training; only generate plots from existing logs")
    args = parser.parse_args()

    if args.debug:
        CFG.debug = True; CFG.epochs = 3; CFG.batch_size = 4
    if args.epochs:      CFG.epochs = args.epochs
    if args.batch_size:  CFG.batch_size = args.batch_size
    if args.lr:          CFG.lr = args.lr
    if args.num_workers: CFG.num_workers = args.num_workers
    if args.output_dir:  CFG.output_dir = args.output_dir
    if args.data_dir:
        CFG.data_dir = args.data_dir
        CFG.train_audio_dir = os.path.join(CFG.data_dir, "train_audio")
        CFG.metadata_path   = os.path.join(CFG.data_dir, "train_metadata.csv")
    CFG.gpus = args.gpus

    if args.plot_only:
        plot_results(CFG.output_dir)
        return

    L.seed_everything(CFG.seed, workers=True)

    print(f"\n{'='*70}")
    print("BirdCLEF 2023 — EfficientNet-B7 + GRU  [Lightning + FP16]")
    print(f"{'='*70}")
    print(f"  GPUs:         {CFG.gpus}")
    print(f"  Batch/GPU:    {CFG.batch_size}")
    print(f"  Effective BS: {CFG.batch_size * CFG.gpus}")
    print(f"  Epochs:       {CFG.epochs}")
    print(f"  Precision:    16-mixed (FP16)")
    print(f"  LR:           {CFG.lr}")
    print(f"{'='*70}\n")

    dm = BirdCLEFDataModule(CFG)
    dm.setup()

    model = BirdCLEFModule(
        num_classes=dm.num_classes, cfg=CFG,
        class_to_idx=dm.class_to_idx, idx_to_class=dm.idx_to_class,
    )

    os.makedirs(CFG.output_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=CFG.output_dir,
            filename="best-{epoch:02d}-{val/acc:.4f}",
            monitor="val/acc", mode="max",
            save_top_k=1, save_last=True, verbose=True,
        ),
        ModelCheckpoint(
            dirpath=CFG.output_dir,
            filename="best-cmap-{epoch:02d}-{val/padded_cmap:.4f}",
            monitor="val/padded_cmap", mode="max",
            save_top_k=1, verbose=True,
        ),
        EarlyStopping(monitor="val/acc", patience=15, mode="max", verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
        TQDMProgressBar(refresh_rate=50),
    ]

    csv_logger = CSVLogger(save_dir=CFG.output_dir, name="lightning_logs")

    strategy = DDPStrategy(find_unused_parameters=True) if CFG.gpus > 1 else "auto"

    trainer = L.Trainer(
        max_epochs=CFG.epochs,
        accelerator="gpu",
        devices=CFG.gpus,
        strategy=strategy,
        precision="16-mixed",
        callbacks=callbacks,
        logger=csv_logger,
        gradient_clip_val=5.0,
        deterministic=False,
        enable_checkpointing=True,
        log_every_n_steps=50,
        val_check_interval=1.0,
        num_sanity_val_steps=2,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)

    print(f"\n{'='*70}")
    print("Running final test evaluation with best checkpoint...")
    print(f"{'='*70}\n")
    trainer.test(model, datamodule=dm, ckpt_path="best")

    if trainer.is_global_zero:
        best_ckpt = trainer.checkpoint_callback.best_model_path
        if best_ckpt and os.path.exists(best_ckpt):
            ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            standalone = {
                "model_state_dict": {
                    k.replace("model.", "", 1): v
                    for k, v in ckpt["state_dict"].items() if k.startswith("model.")
                },
                "num_classes": dm.num_classes,
                "class_to_idx": dm.class_to_idx,
                "idx_to_class": dm.idx_to_class,
                "cfg": {k: v for k, v in vars(CFG).items() if not k.startswith("_")},
            }
            path = os.path.join(CFG.output_dir, "best_model_standalone.pth")
            torch.save(standalone, path)
            print(f"\n[Save] Standalone model weights → {path}")

        plot_results(CFG.output_dir)

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
