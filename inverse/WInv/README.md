# Inverse Wave Problem (WInv)

This directory solves the 1D wave inverse problem using a PINN with learnable wave speed c.

## File structure

- `generate_data.py`: create noisy/clean training set, collocation, boundary, initial, and test splits.
- `train.py`: trains a PINN (via `common.pinn_model.PINN`) on combined losses; saves `wave_inv_model.pth`.
- `finetune.py`: in `main(cfg)`, displays data shapes, loads pretrained model, partitions via `common.partition_data`, prunes (`common.pruning.selective_pruning_multi_layers`), fine-tunes on retained set, evaluates, and saves `wave_inv_finetuned.pth`.
- `README.md`: quickstart and sample config.

## Quick start

```bash

# 1) Train initial model
python -m inverse.WInv.train --config configs/WInv.yaml

# 2) Prune and fine-tune
python -m inverse.WInv.finetune --config configs/WInv.yaml
```