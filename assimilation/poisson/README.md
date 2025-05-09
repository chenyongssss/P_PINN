# Poisson Equation Data Assimilation

This folder implements the pipeline for

```math
\Delta u(x,y) + f(x,y) = 0,\quad (x,y)\in (0,1)^2
```

Quickstart
```bash
python -m assimilation.poisson.train    --config configs/Poisson.yaml
python -m assimilation.poisson.finetune --config configs/Poisson.yaml

```
After running, you’ll have:
model.pth — pretrained weights
finetuned_model.pth — final pruned and fine-tuned weights