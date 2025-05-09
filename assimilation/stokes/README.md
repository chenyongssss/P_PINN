# Stokes Equation Data Assimilation

This folder implements the PINN + selective-prune + fine-tune pipeline for


$$ -\Delta \mathbf u + \nabla p = 0, \quad \nabla\!\cdot\mathbf u = 0,\quad (x,y)\in(0,1)^2 $$



---



Quickstart
```bash
python -m assimilation.stokes.train    --config configs/Stokes.yaml
python -m assimilation.stokes.finetune --config configs/Stokes.yaml

```
After running, you’ll have:
model.pth — pretrained weights
finetuned_model.pth — final pruned and fine-tuned weights