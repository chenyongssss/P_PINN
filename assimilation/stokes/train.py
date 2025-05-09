"""
train.py

1. Adam pretraining on observational data only
2. Compute adaptive weights λ_r, λ_d
3. L-BFGS refinement on composite loss
4. saves stokes_model.pth
"""
import argparse
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from .generate_data import generate_data
import sys
import os

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

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(project_root)


from common.pinn_model import PINN
from common.evaluate import mse
from common.partition_data import partition_data
from common.pruning import selective_pruning_multi_layers

def compute_stokes_residual(model, x):
    """
    Compute the residual for the Stokes equations:
      Momentum equations: -Δu + ∇p = 0 (components: r1, r2)
      Continuity equation: div(u) = 0
    Input x: (N,2) tensor (requires_grad should be True)
    Returns: r1, r2, and divergence of u (each as an (N,1) tensor).
    """
    out = model(x)  # Output (u1, u2, p)
    u = out[:, 0:2]
    p = out[:, 2:3]
    
    # For u1, compute second derivatives
    u1 = u[:, 0:1]
    grads_u1 = torch.autograd.grad(u1, x, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
    u1_x = grads_u1[:, 0:1]
    u1_y = grads_u1[:, 1:2]
    u1_xx = torch.autograd.grad(u1_x, x, grad_outputs=torch.ones_like(u1_x), create_graph=True)[0][:, 0:1]
    u1_yy = torch.autograd.grad(u1_y, x, grad_outputs=torch.ones_like(u1_y), create_graph=True)[0][:, 1:2]
    
    # For u2, compute second derivatives
    u2 = u[:, 1:2]
    grads_u2 = torch.autograd.grad(u2, x, grad_outputs=torch.ones_like(u2), create_graph=True)[0]
    u2_x = grads_u2[:, 0:1]
    u2_y = grads_u2[:, 1:2]
    u2_xx = torch.autograd.grad(u2_x, x, grad_outputs=torch.ones_like(u2_x), create_graph=True)[0][:, 0:1]
    u2_yy = torch.autograd.grad(u2_y, x, grad_outputs=torch.ones_like(u2_y), create_graph=True)[0][:, 1:2]
    
    # Compute pressure gradient
    grads_p = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_x = grads_p[:, 0:1]
    p_y = grads_p[:, 1:2]
    
    # Residuals for momentum equations
    r1 = - (u1_xx + u1_yy) + p_x
    r2 = - (u2_xx + u2_yy) + p_y
    
    # Residual for continuity (divergence)
    div_u = u1_x + u2_y
    
    return torch.cat([r1, r2, div_u], dim=1)

def compute_adaptive_weights(model, collocation_pts, data_pts, u_data, eps=1e-3, lambda_d_max=1e3):
    """
    Compute adaptive weights:
      PDE loss: sum of residuals from the momentum equations and continuity.
      Data loss: velocity error (first two components).
    """
    mse_loss = nn.MSELoss()
    # PDE loss
    r = compute_stokes_residual(model, collocation_pts)
    loss_pde = mse_loss(r, torch.zeros_like(r)) 
    grads_r = torch.autograd.grad(loss_pde, model.parameters(), retain_graph=True, allow_unused=True)
    trace_r = sum(torch.sum(g**2) if g is not None else 0.0 for g in grads_r)
    N_r = collocation_pts.shape[0]
    
    # Data loss (only velocity)
    u_data_pred = model(data_pts)[:, :2]
    loss_data = mse_loss(u_data_pred, u_data)
    grads_d = torch.autograd.grad(loss_data, model.parameters(), retain_graph=True, allow_unused=True)
    trace_d = sum(torch.sum(g**2) if g is not None else 0.0 for g in grads_d)
    N_d = data_pts.shape[0]
    
    R = (trace_r / N_r) + (trace_d / N_d)
    lambda_r = (N_r * R) / (trace_r + eps)
    lambda_d = (N_d * R) / (trace_d + eps)
    
    if lambda_d > lambda_d_max:
        lambda_d = torch.tensor(lambda_d_max, device=collocation_pts.device)
    
    print(f"Adaptive weights computed: lambda_r={lambda_r.item():.6e}, lambda_d={lambda_d.item():.6e}")
    return lambda_r, lambda_d

def main(cfg_path):
    # load config & data
    # Set device: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    seed = 42  # Default seed
    
    # load config
    cfg = yaml.safe_load(open(cfg_path))
    
    # Use seed from config if available
    if 'seed' in cfg:
        seed = cfg['seed']
    
    # Set all seeds
    set_seed(seed)
    
    data = generate_data(cfg)
    x_obs,u_obs = data['x_obs'], data['u_obs']
    x_pde       = data['x_pde']

    # Move data to device
    x_obs = data['x_obs'].to(device).requires_grad_(True)
    u_obs = data['u_obs'].to(device)
    x_pde = data['x_pde'].to(device).requires_grad_(True)

    # build & pretrain
    model = PINN(**cfg['model']).to(device)
    model.train()
    opt = optim.Adam(model.parameters(), lr=cfg['training']['lr_pretrain'], betas=(0.9, 0.999))
    mse_loss = nn.MSELoss()
    for ep in range(1, cfg['training']['pretrain_epochs']+1):
        opt.zero_grad()
        u_pred = model(x_obs)[:, :2]  # only velocity
        loss = mse_loss(u_pred, u_obs)
        loss.backward()
        opt.step()
        if ep % 1000 == 0:
            print(f"[Adam] Epoch {ep}, MSE={loss:.2e}")

    # adaptive weights
    lambda_r, lambda_d = compute_adaptive_weights(
        model, x_pde, x_obs, u_obs,
        eps=cfg['training']['adaptive_eps'],
        lambda_d_max=cfg['training']['lambda_d_max']
    )

    # L-BFGS composite refine
    opt_lb = optim.LBFGS(model.parameters(),
                        max_iter=cfg['training']['lbfgs_max_iter'],history_size=50,
                                  tolerance_grad=1e-9, tolerance_change=1e-9, line_search_fn="strong_wolfe")
    def closure():
        opt_lb.zero_grad()
        r = compute_stokes_residual(model, x_pde)
        loss_pde = mse_loss(r, torch.zeros_like(r))
        u_pred = model(x_obs)[:, :2]  # only velocity
        loss_data = mse_loss(u_pred, u_obs)
        loss = lambda_r*loss_pde+lambda_d*loss_data
        loss.backward()
        return loss
    opt_lb.step(closure)
    print("L-BFGS pretraining complete.")

    torch.save(model.state_dict(), 'stokes_model.pth')
    print("Saved pretrained model to 'stokes_model.pth'.")

    
    print(f"Pre-prune MSE: {mse(model(x_obs)[:,:2],u_obs):.2e}")
    # print(f"Post-prune data MSE: {mse(pruned,x_obs,u_obs):.2e}")

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
