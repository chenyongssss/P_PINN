# Inverse Euler–Bernoulli Beam Problem (EInv)

This folder implements a PINN-based inverse solver for the Euler–Bernoulli beam PDE:

$$ u_{tt} + \alpha^2 u_{xxxx} = 0, \quad u(0,t)=u(1,t)=0, \quad u(x,0)=\sin(\pi x). $$

We cover data generation, training, selective pruning, and fine-tuning.

## Files

- `generate_data.py`: create noisy and clean measurements, collocation, boundary, initial, and test sets.
- `train.py`: defines residual, loss, and trains `PINN` with a learnable `alpha`; saves `beam_inv_model.pth`.
- `finetune.py`: partitions data by composite error (`common.partition_data`), prunes (`common.pruning`), fine-tunes, and evaluates; saves `beam_inv_finetuned.pth`.
- `README.md`: this file.

## Quick start

```bash

# 1) Train initial model
python -m inverse.EInv.train --config configs/EInv.yaml

# 2) Prune and fine-tune
python -m inverse.EInv.finetune --config configs/EInv.yaml
```