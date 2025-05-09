# Inverse Heat Problem (HInv)

This directory contains a PINN-based inverse solver for a heat equation with spatially-varying conductivity a(x,y).
We include initial training, selective pruning, and fine-tuning.

## Files

- `generate_data.py`: generate noisy measurements, PDE collocation points, boundary and initial condition points.
- `train.py`: trains a generic `PINN` model on combined data + PDE + boundary + initial losses; saves `h_inv_model.pth`.
- `finetune.py`: partitions measurement data by mixed error threshold using `common.partition_data`, prunes neurons via `common.pruning.selective_pruning_multi_layers`, fine-tunes pruned model on retained subset; saves `h_inv_finetuned.pth`.
- `README.md`: instructions and sample config.

## Quick Start

```bash

# 1) Train initial model
python -m inverse.HInv.train --config configs/HInv.yaml

# 2) Prune and fine-tune
python -m inverse.HInv.finetune --config configs/HInv.yaml
```