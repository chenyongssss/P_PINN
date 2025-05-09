"""
finetune.py

Fine‐tune the pruned PINN on the 'retain' subset + PDE + BC/IC.
Saves final weights to `finetuned_model.pth`.
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
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
from .train import heat_residual, source_term_heat, heat_residual_with_source

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(project_root)

from common.pinn_model import PINN
from common.partition_data import partition_data
from common.evaluate import mse
from common.pruning import selective_pruning_multi_layers
# change the pruning strategies or pruning criteria
from script.single_step import selective_pruning_multi_layers_single_step
from script.prune_criteria import selective_pruning_multi_layers_abs
from script.prune_criteria import selective_pruning_multi_layers_freq
from script.prune_criteria import selective_pruning_multi_layers_rms
from script.prune_criteria import selective_pruning_multi_layers_std


def fine_tune_heat_pruned(
    model: nn.Module,
    X_good: torch.Tensor,
    u_good: torch.Tensor,
    x_pde: torch.Tensor,
    x_bndry: torch.Tensor,
    source_term_fn,
    heat_residual_fn,
    ft_cfg: dict
) -> nn.Module:
    """
    Two-stage fine-tuning for the pruned heat-PINN:
      1) Adam on data + PDE residual + boundary loss
      2) L-BFGS refinement on the same composite loss

    Args:
      model:             Pruned heat-PINN model.
      X_good, u_good:    Retained high-quality observations.
      x_pde:             Collocation points for PDE loss.
      x_bndry:           Boundary points for BC loss.
      source_term_fn:    Function(x) → source term f(x).
      heat_residual_fn:  Function(model, x) → (u_t, u_xx).
      ft_cfg:            Fine-tuning config dict with keys:
                         'lr_adam', 'adam_epochs',
                         'candidate_lambda_r', 'lbfgs_iter'.
    """
    mse = nn.MSELoss()

    # Stage 1: Adam optimization
    opt_adam = optim.Adam(
        model.parameters(),
        lr=ft_cfg['lr_adam'],
        betas=(0.9, 0.999)
    )
    for ep in range(1, ft_cfg['adam_epochs'] + 1):
        opt_adam.zero_grad()
        # data loss
        pred = model(X_good)
        data_loss = mse(pred, u_good)
        # PDE residual loss
        u_t, u_xx = heat_residual_fn(model, x_pde)
        f = source_term_fn(x_pde)
        pde_loss = mse(u_t - u_xx - f, torch.zeros_like(f))
        # boundary loss (e.g., Dirichlet)
        b_pred = model(x_bndry)
        b_loss = mse(b_pred, torch.zeros_like(b_pred))
        # lambda_r may be list
        lam_r = ft_cfg['candidate_lambda_r']
        if isinstance(lam_r, list):
            lam_r = lam_r[0]
        # composite loss
        loss = data_loss + lam_r * pde_loss + b_loss
        loss.backward()
        opt_adam.step()

        if ep % 500 == 0:
            print(f"[Adam FT] Epoch {ep:4d} | Loss={loss.item():.2e}")

    # Stage 2: L-BFGS refinement
    opt_lbfgs = optim.LBFGS(
        model.parameters(),
        max_iter=ft_cfg['lbfgs_iter'],
        history_size=50,
        tolerance_grad=1e-9,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe"
    )

    def closure():
        opt_lbfgs.zero_grad()
        pred = model(X_good)
        data_loss = mse(pred, u_good)
        u_t, u_xx = heat_residual_fn(model, x_pde)
        f = source_term_fn(x_pde)
        pde_loss = mse(u_t - u_xx - f, torch.zeros_like(f))
        b_pred = model(x_bndry)
        b_loss = mse(b_pred, torch.zeros_like(b_pred))
        lam_r = ft_cfg['candidate_lambda_r']
        if isinstance(lam_r, list):
            lam_r = lam_r[0]
        loss = data_loss + lam_r * pde_loss + b_loss
        loss.backward()
        return loss

    opt_lbfgs.step(closure)
    print("L-BFGS fine-tuning complete.")
    return model
def main(cfg_path):
    # Set device: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # load config
    cfg = yaml.safe_load(open(cfg_path))
    
    # Use seed from config if available
    seed = 42  # Default seed
    if 'seed' in cfg:
        seed = cfg['seed']
    
    # Set all seeds
    set_seed(seed)
    
    # load data
    data = generate_data(cfg)
    
    # Move data to device
    x_obs = data['x_obs'].to(device).requires_grad_(True)
    u_obs = data['u_obs'].to(device)
    x_pde = data['x_pde'].to(device).requires_grad_(True)
    x_bndry = data['x_bndry'].to(device).requires_grad_(True)
    x_test = data['x_test'].to(device).requires_grad_(True)
    u_test = data['u_test'].to(device)

    # rebuild model
    model = PINN(**cfg['model']).to(device)
    
    # Load pruned model weights
    try:
        # Try loading the newer format with metadata
        checkpoint = torch.load('heat_model.pth', map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded pruned model with metadata")
        else:
            # Fall back to old format (direct state dict)
            model.load_state_dict(checkpoint)
            print("Loaded pruned model (legacy format)")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the pruned model was saved correctly")
        return
    
    print("baseline-pinn:")
    init_mse = mse(model(x_test), u_test)
    print(f"Final MSE on all test points: {init_mse:.2e}")
     # selective pruning
    retain, forget = partition_data(
        x_obs, u_obs, model,
        pde_residual_fn = heat_residual_with_source,
        eps=cfg['partition']['eps'],
        w_data=cfg['partition']['w_data'],
        w_res=cfg['partition']['w_res']
    )
    X_good, u_good = retain
    X_bad,  u_bad  = forget

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

    # Call the extracted fine-tuning function
    model = fine_tune_heat_pruned(
        model,
        X_good, u_good,
        x_pde, x_bndry,
        source_term_heat,
        heat_residual,
        ft_cfg=cfg['fine_tune_pruned']
    )

    # final evaluation
    final_mse = mse(model(x_test), u_test)
    print(f"Final MSE on all test points: {final_mse:.2e}")

    # Save the finetuned model with metadata
    finetuned_save_dict = {
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'seed': seed,
        'device': str(device),
        'final_mse': final_mse
    }
    torch.save(finetuned_save_dict, 'heat_finetuned_model.pth')
    print("Saved fine‐tuned model to 'heat_finetuned_model.pth' with metadata.")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))  # P_PINN/assimilation/heat
    project_root = os.path.dirname(os.path.dirname(current_dir))  # P_PINN
    cfg_path = os.path.join(project_root, "configs", "Heat.yaml")
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str,
                   default=cfg_path,
                   help="path to Heat.yaml")
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