# Wave Equation Data Assimilation Example


**PDE 形式：**
```math
u_{tt} = u_{xx},\quad x\in[0,1],\ t\in[0,T_{\rm final}],
\quad u(0,t)=u(1,t)=0,\; u(x,0)=\sin(2\pi x).
```

Quickstart
```bash
python -m assimilation.wave.train    --config configs/Wave.yaml
python -m assimilation.wave.finetune --config configs/Wave.yaml

```
After running, you’ll have:
model.pth — pretrained weights
finetuned_model.pth — final pruned and fine-tuned weights