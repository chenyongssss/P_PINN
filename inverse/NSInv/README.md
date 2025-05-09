# Inverse Navier–Stokes Problem (NS_INV)

This directory implements a PINN-based inverse solver for 2D Navier–Stokes, with selective neuron pruning.

## Files

- `generate_data.py`: load CFD data, split train/test, and generate PDE collocation points.
- `train.py`: defines `InverseNSNet` (inherits from `common.pinn_model.PINN` with `pde_params={'beta1', 'beta2'}`), PDE residual, loss, and trains initial model; saves `ns_inv_model.pth`.
- `finetune.py`: partitions data via composite error, applies pruning (`common.pruning.selective_pruning_multi_layers`), fine-tunes on retained data, evaluates on test set, and saves `ns_inv_finetuned.pth`.

## Quick start

```bash

# 1) Train initial model
python -m inverse.NSInv.train --config configs/NSInv.yaml

# 2) Prune and fine-tune
python -m inverse.NSInv.finetune --config configs/NSInv.yaml
```