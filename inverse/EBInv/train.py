import argparse
import os
import sys
import random
import yaml
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

# allow common imports
current = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(os.path.dirname(current)))
sys.path.append(root)

from common.pinn_model import PINN
from .generate_data import generate_data

# 1. Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# 2. PDE residual: u_tt + alpha^2 * u_xxxx
def beam_residual(model: PINN, X: torch.Tensor) -> torch.Tensor:
    X_req = X.clone().detach().requires_grad_(True)
    u = model(X_req)
    # time derivatives
    u_t = torch.autograd.grad(u, X_req, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,1:2]
    u_tt = torch.autograd.grad(u_t, X_req, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:,1:2]
    # spatial derivatives
    u_x = torch.autograd.grad(u, X_req, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,0:1]
    u_xx = torch.autograd.grad(u_x, X_req, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,0:1]
    u_xxx = torch.autograd.grad(u_xx, X_req, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0][:,0:1]
    u_xxxx = torch.autograd.grad(u_xxx, X_req, grad_outputs=torch.ones_like(u_xxx), create_graph=True)[0][:,0:1]
    alpha = model.pde_params['alpha']
    return u_tt + (alpha**2)*u_xxxx

# 3. Combined loss: data + PDE + BC + IC

def loss_fn(model, X_data, u_data, X_pde, X_b, u_b, X_i, u_i,
                 lambda_pde=1.0, lambda_b=1.0, lambda_i=1.0, lambda_b_extra=1.0, lambda_i_extra=1.0):
    mse = nn.MSELoss()

    # Data loss: Fit the measured data
    u_pred_data = model(X_data)
    loss_data = mse(u_pred_data, u_data)

    # PDE residual loss
    r_pde = beam_residual(model, X_pde)
    loss_pde = mse(r_pde, torch.zeros_like(r_pde))

    # Boundary loss: u=0 at x=0 and x=1
    u_pred_b = model(X_b)
    loss_b = mse(u_pred_b, u_b)
    # Additional boundary constraint: Compute u_xx at boundary (should be 0)
    X_b_in = X_b.clone().detach().requires_grad_(True)
    u_b_pred = model(X_b_in)
    grad_u_b = torch.autograd.grad(u_b_pred, X_b_in, grad_outputs=torch.ones_like(u_b_pred), create_graph=True)[0]
    u_b_x = grad_u_b[:, 0:1]
    u_b_xx = torch.autograd.grad(u_b_x, X_b_in, grad_outputs=torch.ones_like(u_b_x), create_graph=True)[0][:, 0:1]
    loss_b_extra = mse(u_b_xx, torch.zeros_like(u_b_xx))

    # Initial condition: u(x,0)= sin(pi*x)
    u_pred_i = model(X_i)
    loss_i = mse(u_pred_i, u_i)
    # Additional initial condition constraint: u_t at t=0 should be 0
    X_i_in = X_i.clone().detach().requires_grad_(True)
    u_i_pred = model(X_i_in)
    grad_u_i = torch.autograd.grad(u_i_pred, X_i_in, grad_outputs=torch.ones_like(u_i_pred), create_graph=True)[0]
    u_i_t = grad_u_i[:, 1:2]
    loss_i_extra = mse(u_i_t, torch.zeros_like(u_i_t))

    total_loss = loss_data + lambda_pde*loss_pde + lambda_b*(loss_b + lambda_b_extra*loss_b_extra) \
                 + lambda_i*(loss_i + lambda_i_extra*loss_i_extra)
    return total_loss, loss_data, loss_pde, loss_b, loss_b_extra, loss_i, loss_i_extra

# 4. Training loop
def train_model(model, cfg, X_data, u_data, X_pde, X_b, u_b, X_i, u_i,
                epochs=20000, lr=1e-3, 
                lambda_pde=1.0, lambda_b=1.0, lambda_i=1.0,
                lambda_b_extra=1.0, lambda_i_extra=1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    for epoch in range(epochs):
        optimizer.zero_grad()
        L, L_data, L_pde, L_b, L_b_extra, L_i, L_i_extra = loss_fn(
            model, X_data, u_data, X_pde, X_b, u_b, X_i, u_i,
            lambda_pde, lambda_b, lambda_i, lambda_b_extra, lambda_i_extra)
        L.backward()
        optimizer.step()
        if epoch % 2000 == 0:
            print(f"Epoch {epoch}: total={L.item():.4e}, data={L_data.item():.4e}, pde={L_pde.item():.4e}, "
                  f"bdy={L_b.item():.4e}, bdy_extra={L_b_extra.item():.4e}, init={L_i.item():.4e}, init_extra={L_i_extra.item():.4e}")

    # save
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, 'beam_inv_model.pth')
    print("Saved initial beam_inv_model.pth")

# 5. Network class inheriting PINN
def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg.get('seed',2))
    data = generate_data(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # move data to device
    Xd, ud = data['x_obs'].to(device), data['u_obs'].to(device)
    Xp = data['X_pde'].to(device)
    Xb, ub = data['X_b'].to(device), data['u_b'].to(device)
    Xi, ui = data['X_i'].to(device), data['u_i'].to(device)
    # Xt, ut = data['x_test'].to(device), data['u_test'].to(device)
    # build model
    md = cfg['model']
    model = PINN(in_dim=2,
                 hidden_dim=md['hidden_dim'],
                 hidden_layers=md['hidden_layers'],
                 out_dim=md['out_dim'],
                 pde_params={'alpha': md.get('init_alpha',0.0)})
    model.to(device).train()
    train_model(model, cfg, Xd, ud, Xp, Xb, ub, Xi, ui, cfg['training']['epochs'], cfg['training']['lr'],
                cfg['training']['lambda_pde'], cfg['training']['lambda_b'], cfg['training']['lambda_i'], 
                cfg['training']['lambda_b1'], cfg['training']['lambda_i1'])

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))  # P_PINN/inverse/einv
    project_root = os.path.dirname(os.path.dirname(current_dir))  # P_PINN
    cfg_path = os.path.join(project_root, "configs", "EInv.yaml")
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str,
                   default=cfg_path,
                   help="path to EInv.yaml")
    p.add_argument('--seed', type=int, 
                   default=42,
                   help="random seed for reproducibility")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    with open(args.config, 'w') as f:
        yaml.dump(cfg, f)
    main(args.config)