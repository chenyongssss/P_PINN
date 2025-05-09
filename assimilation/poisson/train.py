"""
train.py

Pretrain a PINN on Poisson data, compute adaptive weights,
refine with L‐BFGS, then perform selective pruning.

Saves:
  - pretrained model to 'model.pth'
  - pruned model to      'pruned_model.pth'
"""
import argparse
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from .generate_data import generate_data, source_term
import random
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
from common.evaluate     import mse
from common.partition_data import partition_data
from common.pruning      import selective_pruning_multi_layers
from common.pinn_model import PINN
from common.evaluate import mse
from common.partition_data import partition_data
from common.pruning import selective_pruning_multi_layers

def compute_laplacian(u, x):
    """
    Compute the Laplacian of u with respect to x.
    u: Tensor of shape (N,1) (output)
    x: Tensor of shape (N,2) (input), with x.requires_grad=True.
    Returns: Tensor of shape (N,1)
    """
    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    grad_u_x = grad_u[:, 0:1]
    grad_u_y = grad_u[:, 1:2]
    grad_u_xx = torch.autograd.grad(grad_u_x, x, grad_outputs=torch.ones_like(grad_u_x), create_graph=True)[0][:, 0:1]
    grad_u_yy = torch.autograd.grad(grad_u_y, x, grad_outputs=torch.ones_like(grad_u_y), create_graph=True)[0][:, 1:2]
    return grad_u_xx + grad_u_yy

def poisson_residual(model, x):
    """
    PDE residual Δu + f = 0.
    """
    u = model(x)
    lap = compute_laplacian(u, x)
    f = source_term(x)
    return lap + f

#. Adaptive Weight Computation Function (using PDE and data loss)
########################################################
def compute_adaptive_weights(model, collocation_pts, data_pts, u_data, eps=1e-3, lambda_d_max=1e3):
    mse_loss = nn.MSELoss()
    # PDE loss
    u_coll = model(collocation_pts)
    lap_u = compute_laplacian(u_coll, collocation_pts)
    f = source_term(collocation_pts)
    loss_pde = mse_loss(lap_u + f, torch.zeros_like(f))
    grads_r = torch.autograd.grad(loss_pde, model.parameters(), retain_graph=True, allow_unused=True)
    trace_r = sum(torch.sum(g**2) if g is not None else 0.0 for g in grads_r)
    N_r = collocation_pts.shape[0]
    
    # Data loss
    u_data_pred = model(data_pts)
    loss_data = mse_loss(u_data_pred, u_data)
    grads_d = torch.autograd.grad(loss_data, model.parameters(), retain_graph=True, allow_unused=True)
    trace_d = sum(torch.sum(g**2) if g is not None else 0.0 for g in grads_d)
    N_d = data_pts.shape[0]
    
    R = (trace_r / N_r) + (trace_d / N_d)
    lambda_r = (N_r * R) / (trace_r + eps)
    lambda_d = (N_d * R) / (trace_d + eps)
    
    # Clip lambda_d to prevent it from being too large
    if lambda_d > lambda_d_max:
        lambda_d = torch.tensor(lambda_d_max, device=lambda_d.device) if isinstance(lambda_d, torch.Tensor) else lambda_d_max
    
    print(f"Adaptive weights computed: lambda_r={lambda_r.item():.6e}, lambda_d={lambda_d if isinstance(lambda_d, float) else lambda_d.item():.6e}")
    return lambda_r, lambda_d

def main(cfg_path):
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

    # data
    data = generate_data(cfg)
    x_obs = data['x_obs'].to(device).requires_grad_(True)
    u_obs = data['u_obs'].to(device)
    x_pde = data['x_pde'].to(device).requires_grad_(True)

    # x_test, u_test = data['x_test'], data['u_test']

    # model
    model = PINN(**cfg['model']).to(device)
    model.train()

    # 1) Adam pretraining (data only)
    opt = optim.Adam(model.parameters(), lr=cfg['training']['lr_pretrain'], betas=(0.9,0.999))
    mse_loss = torch.nn.MSELoss()
    for ep in range(1, cfg['training']['pretrain_epochs']+1):
        opt.zero_grad()
        pred = model(x_obs)
        loss = mse_loss(pred, u_obs)
        loss.backward()
        opt.step()
        if ep % 1000 == 0:
            print(f"[Adam] Epoch {ep}, Data MSE: {loss:.2e}")

    
    lambda_r, lambda_d = compute_adaptive_weights(
        model, x_pde, x_obs, u_obs,
        eps=cfg['training']['adaptive_eps'],
        lambda_d_max=cfg['training']['lambda_d_max']
    )  

    # 3) L‐BFGS refinement
    opt_lb = optim.LBFGS(model.parameters(),
                        max_iter=cfg['training']['lbfgs_max_iter'],
                        history_size=50, tolerance_grad=1e-9, tolerance_change=1e-9, line_search_fn="strong_wolfe")
    def closure():
        opt_lb.zero_grad()
        u_coll = model(x_pde)
        lap = compute_laplacian(u_coll, x_pde)
        f   = source_term(x_pde)
        loss_pde = mse_loss(lap + f, torch.zeros_like(f))
        pred = model(x_obs)
        loss_data = mse_loss(pred, u_obs)
        loss = lambda_r * loss_pde + lambda_d * loss_data
        loss.backward()
        return loss

    opt_lb.step(closure)
    print("L-BFGS pretraining complete.")

    # save pretrained
    torch.save(model.state_dict(), 'poisson_model.pth')
    print("Saved pretrained model to 'poisson_model.pth'.")


if __name__=='__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))  # P_PINN/assimilation/poisson
    project_root = os.path.dirname(os.path.dirname(current_dir))  # P_PINN
    cfg_path = os.path.join(project_root, "configs", "Poisson.yaml")
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str,
                   default=cfg_path,
                   help="path to Poisson.yaml")
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