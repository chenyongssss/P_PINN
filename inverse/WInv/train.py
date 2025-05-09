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
from common.evaluate import l2_relative_error
from .generate_data import generate_data

# Set reproducible seeds
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# PDE residual for wave: u_tt - c^2 u_xx
def wave_residual(model: PINN, X: torch.Tensor) -> torch.Tensor:
    X_req = X.clone().detach().requires_grad_(True)
    u = model(X_req)
    grad = torch.autograd.grad(u, X_req, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = grad[:,1:2]
    u_tt = torch.autograd.grad(u_t, X_req, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:,1:2]
    u_x = grad[:,0:1]
    u_xx = torch.autograd.grad(u_x, X_req, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,0:1]
    c = model.pde_params['c']
    return u_tt - (c**2)*u_xx

# Combined loss: data + PDE + BC + IC

def loss_fn(model, X_data, u_data, X_pde, X_b, u_b, X_i, u_i,
                 lambda_r=1.0, lambda_b=1.0, lambda_i1=1.0, lambda_i2=1.0, lambda_d=1.0):
    mse = nn.MSELoss()
    # Data loss: fit the observed data u
    u_pred_data = model(X_data)
    loss_d = mse(u_pred_data, u_data)
    # PDE residual loss
    r_pde = wave_residual(model, X_pde)
    loss_r = mse(r_pde, torch.zeros_like(r_pde))
    # Boundary condition: u(0,t)=u(1,t)=0
    u_pred_b = model(X_b)
    loss_b = mse(u_pred_b, u_b)
    # Initial condition: u(x,0)= sin(pi*x)+0.5*sin(4*pi*x)
    u_pred_i = model(X_i)
    loss_i1 = mse(u_pred_i, u_i)
    # Additional initial condition constraint: u_t(x,0)=0
    X_i_in = X_i.clone().detach().requires_grad_(True)
    u_i_pred = model(X_i_in)
    grad_u_i = torch.autograd.grad(u_i_pred, X_i_in, grad_outputs=torch.ones_like(u_i_pred), create_graph=True)[0]
    u_i_t = grad_u_i[:, 1:2]
    loss_i2 = mse(u_i_t, torch.zeros_like(u_i_t))
    
    total_loss = lambda_d * loss_d + lambda_r * loss_r + lambda_b * loss_b + lambda_i1 * loss_i1 + lambda_i2 * loss_i2
    return total_loss, loss_d, loss_r, loss_b, loss_i1, loss_i2

def evaluate_model_data_wave(model, X_test, u_test, true_c=2.0):
    model.eval()
    with torch.no_grad():
        u_pred = model(X_test)
    
    
    rel_u = l2_relative_error(u_pred,u_test)
    pred_c = model.pde_params['c']
    c_err = abs(pred_c - true_c)
    print(f'the mse of c: {c_err}')
    return rel_u, c_err
# Training loop

def train_model(model, cfg, X_data, u_data, X_pde, X_b, u_b, X_i, u_i,
                      epochs=20000, lr=1e-3,
                      lambda_r=1.0, lambda_b=1.0, lambda_i1=1.0, lambda_i2=1.0, lambda_d=1.0):
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'], betas=(0.9, 0.999))
    for epoch in range(epochs):
        optimizer.zero_grad()
        L, L_d, L_r, L_b, L_i1, L_i2 = loss_fn(model, X_data, u_data, X_pde, X_b, u_b, X_i, u_i,
                                                     lambda_r, lambda_b, lambda_i1, lambda_i2, lambda_d)
        L.backward()
        optimizer.step()
        if epoch % 2000 == 0:
            print(f"Epoch {epoch}: total={L.item():.4e}, data={L_d.item():.4e}, pde={L_r.item():.4e}, "
                  f"boundary={L_b.item():.4e}, init={L_i1.item():.4e}, init_t={L_i2.item():.4e}")
    torch.save({'model_state_dict': model.state_dict(),'config':cfg}, 'wave_inv_model.pth')
    print("Saved wave_inv_model.pth")
# Entry point
def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    set_seed(cfg.get('seed',0))
    # generate data
    data = generate_data(cfg)
    # move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xd, ud = data['X_data'].to(device), data['u_data'].to(device)
    Xp = data['X_pde'].to(device)
    Xb, ub = data['X_b'].to(device), data['u_b'].to(device)
    Xi, ui = data['X_i'].to(device), data['u_i'].to(device)
    Xt, ut = data['X_i'].to(device), data['u_i'].to(device)
    
    # build model with learnable c
    md=cfg['model']
    model = PINN(
        in_dim=2,
        hidden_dim=md['hidden_dim'],
        hidden_layers=md['hidden_layers'],
        out_dim=1,
        pde_params={'c': md.get('init_c',1.0)}
    ).to(device)
    train_model(model, cfg, Xd, ud, Xp, Xb, ub, Xi, ui, cfg['training']['epochs'], cfg['training']['lr'],
                cfg['training']['lambda_pde'], cfg['training']['lambda_b'], cfg['training']['lambda_i1'], 
                cfg['training']['lambda_i2'], cfg['training']['lambda_d'])
    evaluate_model_data_wave(model, Xt, ut)

if __name__=='__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))  # P_PINN/inverse/winv
    project_root = os.path.dirname(os.path.dirname(current_dir))  # P_PINN
    cfg_path = os.path.join(project_root, "configs", "WInv.yaml")
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