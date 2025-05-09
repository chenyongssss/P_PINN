# Inverse Poisson Problem (Poisson_INV)

This folder implements a Physics-Informed Neural Network (PINN) to solve the inverse Poisson problem with selective pruning.

## File overview

- `generate_data.py`: generate noisy observations, collocation points, boundary points, and high-resolution test grid (functions: `generate_data(cfg)`).
- `train.py`: initial training script using Adam optimizer on combined data + PDE + boundary losses; saves `poisson_inv_model.pth`.
- `finetune.py`: loads pre-trained model, partitions data into retain/forget sets via `common.partition_data`, applies selective pruning (`common.pruning.selective_pruning_multi_layers`), fine-tunes pruned model (Adam + L-BFGS), evaluates on test grid, and saves `poisson_inv_finetuned.pth`.

## Quick start

```bash
# 1) Train initial model
python -m inverse.PInv.train --config configs/PInv.yaml

# 2) Prune and fine-tune
python -m inverse.PInv.finetune --config configs/PInv.yaml
```