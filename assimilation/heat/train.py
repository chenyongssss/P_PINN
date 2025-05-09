"""
train.py

Pretrain a PINN for heat‐equation data assimilation,
Uses GPU if available and fixes random seed for reproducibility.
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
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
from common.evaluate     import mse
from common.partition_data import partition_data
from common.pruning      import selective_pruning_multi_layers


def source_term_heat(x):
    """
    Source term f(x,t) = u_t - u_xx for u(x,t)=exp(-4*pi^2*t^2)*sin(2*pi*x):
      u_t = -8*pi^2*t*exp(-4*pi^2*t^2)*sin(2*pi*x)
      u_xx = -4*pi^2*exp(-4*pi^2*t^2)*sin(2*pi*x)
    Thus, f = 4*pi^2*exp(-4*pi^2*t^2)*sin(2*pi*x)*(1-2*t)
    Returns a (N,1) tensor.
    """
    x_val = x[:, 0:1]
    t_val = x[:, 1:2]
    f = 4*(np.pi**2)*torch.exp(-4*(np.pi**2)*t_val**2) * torch.sin(2*np.pi*x_val) * (1 - 2*t_val)
    return f

def heat_residual(model, x):
    """
    u_t , u_xx  for the heat equation.
    """
    # first derivatives
    u = model(x)
    grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:,0:1]
    u_t = grads[:,1:2]
    # second derivative
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    return u_t, u_xx

def compute_adaptive_weights_all(model, collocation_pts, data_pts, u_data, boundary_pts, eps=1e-3, lambda_d_max=1e3, lambda_b_max=1e3):
    """
    Compute adaptive weights for the composite loss (PDE, data, and boundary).
    """
    mse_loss = nn.MSELoss()
    # PDE loss
    u_t, u_xx = heat_residual(model, collocation_pts)
    f = source_term_heat(collocation_pts)
    
    loss_pde = mse_loss(u_t - u_xx - f, torch.zeros_like(f))
    grads_r = torch.autograd.grad(loss_pde, model.parameters(), retain_graph=True, allow_unused=True)
    trace_r = sum(torch.sum(g**2) if g is not None else 0.0 for g in grads_r)
    N_r = collocation_pts.shape[0]
    
    # Data loss
    u_data_pred = model(data_pts)
    loss_data = mse_loss(u_data_pred, u_data)
    grads_d = torch.autograd.grad(loss_data, model.parameters(), retain_graph=True, allow_unused=True)
    trace_d = sum(torch.sum(g**2) if g is not None else 0.0 for g in grads_d)
    N_d = data_pts.shape[0]
    
    # Boundary loss
    u_boundary = model(boundary_pts)
    loss_boundary = mse_loss(u_boundary, torch.zeros_like(u_boundary))
    grads_b = torch.autograd.grad(loss_boundary, model.parameters(), retain_graph=True, allow_unused=True)
    trace_b = sum(torch.sum(g**2) if g is not None else 0.0 for g in grads_b)
    N_b = boundary_pts.shape[0]
    
    R = (trace_r / N_r) + (trace_d / N_d) + (trace_b / N_b)
    lambda_r = (N_r * R) / (trace_r + eps)
    lambda_d = (N_d * R) / (trace_d + eps)
    lambda_b = (N_b * R) / (trace_b + eps)
    
    if lambda_d > lambda_d_max:
        lambda_d = torch.tensor(lambda_d_max, device=collocation_pts.device)
    if lambda_b > lambda_b_max:
        lambda_b = torch.tensor(lambda_b_max, device=collocation_pts.device)
    
    print(f"Adaptive weights computed: lambda_r={lambda_r.item():.6e}, lambda_d={lambda_d.item():.6e}, lambda_b={lambda_b.item():.6e}")
    return lambda_r, lambda_d, lambda_b

def heat_residual_with_source(model, x):
    """
    u_t - u_xx - f
    """
    #heat_residual
    u_t, u_xx = heat_residual(model, x)
    
    # source_term_heat
    f = source_term_heat(x)
    
    # u_t - u_xx - f
    residual = u_t - u_xx - f
    
    return residual



def main(cfg_path):
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

    # generate data
    data = generate_data(cfg)
    
    # Move data to device
    x_obs = data['x_obs'].to(device).requires_grad_(True)
    u_obs = data['u_obs'].to(device)
    x_pde = data['x_pde'].to(device).requires_grad_(True)
    x_bndry = data['x_bndry'].to(device).requires_grad_(True)

    # build model and move to device
    model = PINN(**cfg['model']).to(device)
    model.train()

    # Adam pretraining
    mse_loss = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr_pretrain'], betas=(0.9,0.999))
    for epoch in range(1, cfg['training']['pretrain_epochs']+1):
        optimizer.zero_grad()
        u_pred = model(x_obs)
        loss = mse_loss(u_pred, u_obs)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Pretrain Epoch {epoch}, Data Loss: {loss.item():.6f}")
    print("Pretraining (data loss only) completed.")

    # Load adaptive weights
    eps, lambda_d_max, lambda_b_max = cfg['training']['adaptive_eps'], cfg['training']['lambda_d_max'], cfg['training']['lambda_b_max']
    lambda_r, lambda_d, lambda_b = compute_adaptive_weights_all(model, x_pde, x_obs, u_obs, x_bndry, eps, lambda_d_max, lambda_b_max)
    
    # L-BFGS refinement
    optim_lbfgs = optim.LBFGS(model.parameters(),
                              max_iter=cfg['training']['lbfgs_max_iter'],
                              history_size=50, tolerance_grad=1e-9, tolerance_change=1e-9, line_search_fn="strong_wolfe")
    def closure():
        optim_lbfgs.zero_grad()
        pred_obs = model(x_obs)
        data_loss = mse_loss(pred_obs, u_obs)
        u_t, u_xx = heat_residual(model, x_pde)
        f = source_term_heat(x_pde)
        pde_loss = mse_loss(u_t-u_xx-f, torch.zeros_like(f))
        pred_b = model(x_bndry)
        bndry_loss = mse_loss(pred_b, torch.zeros_like(pred_b))
        # lam = cfg['candidate_lambda_r'][0]
        total = lambda_d*data_loss + lambda_r * pde_loss + lambda_b * bndry_loss
        total.backward()
        return total

    optim_lbfgs.step(closure)
    print("L-BFGS pretraining complete.")
    
    # Save the pre-trained model with metadata
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'seed': seed,
        'device': str(device)
    }
    torch.save(save_dict, 'heat_model.pth')
    print("Pre-trained model saved to 'heat_model.pth' with training metadata.")

   

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