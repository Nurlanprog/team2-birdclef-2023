# team2-birdclef-2023

Kaggle BirdCLEF 2023 competition project. Classifies bird species from audio recordings converted to mel spectrograms.

## Models

| Script | Architecture | Framework |
|---|---|---|
| `train_birdclef.py` | EfficientNetB0 + Transformer Encoder | Keras / TensorFlow |
| `train.py` | EfficientNet-B7 + GRU | PyTorch Lightning |

## Scripts

- **`train_birdclef.py`** — Main training script. Saves model weights, history, and diagnostic plots to `--output_dir`.
- **`train.py`** — PyTorch Lightning trainer with multi-GPU (DDP) and FP16 support.
- **`pruning.py`** — Iterative magnitude pruning via the Lottery Ticket Hypothesis (Frankle & Carlin, ICLR 2019).
- **`visualize.py`** — Standalone visualization of saved results (no retraining needed).

## Quick Start

```bash
# Train (Keras)
python train_birdclef.py --dataset_dir <train/> --path_data <img_stats.csv> --output_dir <outputs/>

# Train (PyTorch, 4 GPUs)
python train.py --gpus 4

# Prune
python pruning.py --output_dir <outputs/> --model_path <model.keras> --weights_path <weights.h5> --path_data <img_stats.csv> --dataset_dir <train/>

# Visualize
python visualize.py --output_dir <outputs/> --model_path <model.keras> --path_data <img_stats.csv> --dataset_dir <train/>
```

## Output Structure

```
outputs/
├── model_*.keras
├── weights_*.h5
├── history_*.csv
├── results_*.json
├── plots/          # training curves, F1 distribution, PR curves, etc.
├── pruning/        # per-round pruning results
└── tb/             # TensorBoard logs
```
