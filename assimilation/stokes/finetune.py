"""
finetune.py

Fine-tune the pruned Stokes PINN on retained data + PDE residual:

1. Adam on retained set
2. Short L-BFGS refine
3. Save finetuned_model.pth and report test MSE
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MSELoss
import random
import numpy as np
import os
import sys

# Set seeds for reproducibility
def set_seed(seed=42):
    """
    Set seeds for reproducibility across Python, NumPy and PyTorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")

from .generate_data import generate_data
# reuse residual from train.py
from .train import compute_stokes_residual


current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(project_root)
from common.pinn_model import PINN
from common.evaluate import mse
from common.partition_data import partition_data
from common.pruning import selective_pruning_multi_layers
# change the pruning strategies or pruning criteria
from script.single_step import selective_pruning_multi_layers_single_step
from script.prune_criteria import selective_pruning_multi_layers_abs
from script.prune_criteria import selective_pruning_multi_layers_freq
from script.prune_criteria import selective_pruning_multi_layers_rms
from script.prune_criteria import selective_pruning_multi_layers_std


# for stokes equation, the label contains only the first dimension
def partition_data(x_obs: torch.Tensor,
                   u_obs: torch.Tensor,
                   model: torch.nn.Module,
                   pde_residual_fn: callable = None,
                   eps: float = 0.01,
                   w_data: float = 1.0,
                   w_res: float = 1.0):
    """
    Split observed data into 'retain' vs. 'forget' sets based on composite score:
      score_i = w_data * ||u_pred_i - u_obs_i|| + w_res * ||residual_i||

    Args:
        x_obs (Tensor[N,in_dim]):  input coordinates of observations
        u_obs (Tensor[N,out_dim]): observed solution values
        model (nn.Module):         a PINN instance
        pde_residual_fn (callable):fn(model, x) -> residual Tensor[N, ...]
        eps (float):               threshold to split
        w_data (float):            weight on data‐error term
        w_res (float):             weight on residual‐error term

    Returns:
        retain_data
        forget_data
    """
    model.eval()
    x_obs = x_obs.clone().detach().requires_grad_(True)
    
    u_pred = model(x_obs)
    
    u_pred = u_pred[:,:-1]

    # data‐error norm per sample
    data_err = ((u_pred - u_obs)**2).sum(dim=1)
        
    if pde_residual_fn is not None:
        res = pde_residual_fn(model, x_obs)
        res_err = (res**2).sum(dim=1)
    else:
        res_err = torch.zeros_like(data_err)

    scores = w_data * data_err + w_res * res_err
    scores = scores.detach().cpu().numpy().squeeze()
    
    good_idx = np.where(scores < eps)[0]
    bad_idx = np.where(scores >= eps)[0]
    print(f"Composite loss threshold: {eps:.6e}, Good data count: {len(good_idx)}, Bad data count: {len(bad_idx)}")
    X_good = x_obs[good_idx].detach()
    u_good = u_obs[good_idx]
    X_bad = x_obs[bad_idx].detach()
    u_bad = u_obs[bad_idx]
    return (X_good, u_good), (X_bad, u_bad)


def fine_tune(
    model: nn.Module,
    X_good: torch.Tensor,
    u_good: torch.Tensor,
    x_pde: torch.Tensor,
    compute_pde_residual,
    lr_adam: float,
    adam_epochs: int,
    lambda_d: float,
    lambda_r: float,
    lbfgs_iter: int
) -> nn.Module:
    """
    Perform two‐stage fine-tuning on a pruned PINN model:
      1) Adam optimization on retained data + PDE residual
      2) L-BFGS refinement on the same loss

    Args:
      model:           Pruned PINN model.
      X_good, u_good:  High-quality observation subset.
      x_pde:           Collocation points for PDE residual.
      compute_pde_residual: Function(model, x) -> residual tensor.
      lr_adam:         Learning rate for Adam.
      adam_epochs:     Number of Adam epochs.
      lambda_d:        Weight for data fidelity loss.
      lambda_r:        Weight for PDE residual loss.
      lbfgs_iter:      Maximum iterations for L-BFGS.
    """
    mse_loss = nn.MSELoss()

    # Stage 1: Adam fine-tuning
    optimizer_adam = optim.Adam(model.parameters(), lr=lr_adam, betas=(0.9, 0.999))
    for epoch in range(1, adam_epochs + 1):
        optimizer_adam.zero_grad()
        # Data fidelity loss
        u_pred = model(X_good)[:, :2]
        loss_data = mse_loss(u_pred, u_good)
        # PDE residual loss
        residual = compute_pde_residual(model, x_pde)
        loss_pde = mse_loss(residual, torch.zeros_like(residual))
        # Combined loss
        loss = lambda_d * loss_data + lambda_r * loss_pde
        loss.backward()
        optimizer_adam.step()

        if epoch % 500 == 0:
            print(f"[Adam FT] Epoch {epoch:4d} | Loss: {loss.item():.2e} | "
                  f"Data: {loss_data.item():.2e} | PDE: {loss_pde.item():.2e}")

    # Stage 2: L-BFGS refinement
    optimizer_lb = optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_iter,
        history_size=50,
        tolerance_grad=1e-9,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer_lb.zero_grad()
        u_pred = model(X_good)[:, :2]
        loss_data = mse_loss(u_pred, u_good)
        residual = compute_pde_residual(model, x_pde)
        loss_pde = mse_loss(residual, torch.zeros_like(residual))
        loss = lambda_d * loss_data + lambda_r * loss_pde
        loss.backward()
        return loss

    optimizer_lb.step(closure)
    print("L-BFGS fine-tuning complete.")

    return model


def main(cfg_path: str):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    cfg = yaml.safe_load(open(cfg_path))

    # Set random seed
    seed = cfg.get('seed', 42)
    set_seed(seed)

    # Generate and move data to device
    data = generate_data(cfg)
    x_obs = data['x_obs'].to(device).requires_grad_(True)
    u_obs = data['u_obs'].to(device)
    x_pde = data['x_pde'].to(device).requires_grad_(True)
    x_test = data['x_test'].to(device).requires_grad_(True)
    u_test = data['u_test'].to(device)

    # Rebuild and load pruned model
    model = PINN(**cfg['model']).to(device)
    try:
        checkpoint = torch.load('stokes_model.pth', map_location=device)
        state = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state)
        print("Loaded pruned model weights.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Evaluate baseline PINN
    init_mse = mse(model(x_test)[:, :2], u_test[:, :2])
    print(f"Baseline PINN initial MSE: {init_mse:.2e}")

    # Partition data into retained (good) and forgotten (bad) subsets
    retain, forget = partition_data(
        x_obs, u_obs, model,
        pde_residual_fn=compute_stokes_residual,
        eps=cfg['partition']['eps'],
        w_data=cfg['partition']['w_data'],
        w_res=cfg['partition']['w_res']
    )
    X_good, u_good = retain

    # Selective pruning
    # selective pruning, strategy and criteria
    strat = cfg['pruning']['strategy']
    if strat == 'rms':
        prune_fn = selective_pruning_multi_layers_rms
    elif strat == 'freq':
        prune_fn = selective_pruning_multi_layers_freq
    elif strat == 'std':
        prune_fn = selective_pruning_multi_layers_std
    elif strat == 'abs':
        prune_fn = selective_pruning_multi_layers_abs
    elif strat == 'single_step':
        prune_fn = selective_pruning_multi_layers_single_step
    else:
        prune_fn = selective_pruning_multi_layers
    if strat!='single_step':
        model = prune_fn(model,
        layer_indices=cfg['pruning']['layers'],
        retain_data=retain,
        forget_data=forget,
        alpha=cfg['pruning']['alpha'],
        num_iter=cfg['pruning']['num_iter'])
    else:
        alpha_single = 1-(1-cfg['pruning']['alpha'])**cfg['pruning']['num_iter']
        model = prune_fn(model, layer_indices=cfg['pruning']['layers'],
        retain_data=retain,
        forget_data=forget,
        alpha=alpha_single)

    # Fine-tune pruned model
    model = fine_tune(
        model,
        X_good, u_good,
        x_pde,
        compute_stokes_residual,
        lr_adam=cfg['fine_tune_pruned']['lr_adam'],
        adam_epochs=cfg['fine_tune_pruned']['adam_epochs'],
        lambda_d=cfg['fine_tune_pruned']['lambda_d'],
        lambda_r=cfg['fine_tune_pruned']['lambda_r'],
        lbfgs_iter=cfg['fine_tune_pruned']['lbfgs_iter']
    )

    # Final evaluation and save
    final_mse = mse(model(x_test)[:, :2], u_test[:, :2])
    print(f"Pruned PINN final MSE: {final_mse:.2e}")
    torch.save(model.state_dict(), 'stokes_finetuned_model.pth')
    print("Saved fine-tuned model to 'stokes_finetuned_model.pth'.")




if __name__=='__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))  # P_PINN/assimilation/heat
    project_root = os.path.dirname(os.path.dirname(current_dir))  # P_PINN
    cfg_path = os.path.join(project_root, "configs", "Stokes.yaml")
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str,
                   default=cfg_path,
                   help="path to Stokes.yaml")
    p.add_argument('--seed', type=int, 
                   default=42,
                   help="random seed for reproducibility")
    args = p.parse_args()
    
    # Override seed in config if provided through command line
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    with open(args.config, 'w') as f:
        yaml.dump(cfg, f)
        
    main(args.config)