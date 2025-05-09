# Heat Equation Data Assimilation Example

This folder shows how to:

1. **Pretrain & Prune**  
2. **Fine‐tune**  

a PINN for the 1D heat‐equation:

```math
u_t = u_{xx},\quad x \in [0,1],\ t \in [0,T_{\rm final}],\ 
u(x,0)=\sin(\pi x),\ u(0,t)=u(1,t)=0.
```

Quickstart
From the repo root:

```bash
python -m assimilation.heat.train   --config configs/Heat.yaml
python -m assimilation.heat.finetune --config configs/Heat.yaml
```
After that you will have:

After running, you’ll have:
model.pth — pretrained weights
finetuned_model.pth — final pruned and fine-tuned weights