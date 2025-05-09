import argparse
import os
import sys
import random
import yaml
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

# allow imports from common
current = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(os.path.dirname(current)))
sys.path.append(root)

from common.pinn_model import PINN
from .generate_data import generate_data


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class InverseNSNet(PINN):
    """
    PINN for 2D Navier–Stokes inverse problem with trainable viscosity params.
    Inherits from common.pinn_model.PINN and uses pde_params for beta1 and beta2.
    Outputs [u,v,p].
    """
    def __init__(self, in_dim=3, hidden_dim=100, hidden_layers=5, out_dim=3):
        # register beta1 and beta2 as learnable PDE parameters
        pde_params = {'beta1': 1.0, 'beta2': 1.0}
        super().__init__(in_dim=in_dim,
                         hidden_dim=hidden_dim,
                         hidden_layers=hidden_layers,
                         out_dim=out_dim,
                         pde_params=pde_params)

# PDE residual for Navier–Stokes

def ns_residual(model, x: torch.Tensor) -> torch.Tensor:
    """
    Compute NS residuals: momentum equations and continuity.
    """
    x_req = x.clone().detach().requires_grad_(True)
    out = model(x_req)
    u, v, p = out[:,0:1], out[:,1:2], out[:,2:3]
    # extract learnable parameters
    b1 = model.pde_params['beta1']
    b2 = model.pde_params['beta2']
    ones = torch.ones_like(u)
    grad = lambda f: torch.autograd.grad(f, x_req, grad_outputs=ones, create_graph=True)[0]
    u_x, u_y, u_t = grad(u).split(1, dim=1)
    v_x, v_y, v_t = grad(v).split(1, dim=1)
    p_x, p_y = grad(p).split(1, dim=1)[:2]
    u_xx = torch.autograd.grad(u_x, x_req, grad_outputs=ones, create_graph=True)[0][:,0:1]
    u_yy = torch.autograd.grad(u_y, x_req, grad_outputs=ones, create_graph=True)[0][:,1:2]
    v_xx = torch.autograd.grad(v_x, x_req, grad_outputs=ones, create_graph=True)[0][:,0:1]
    v_yy = torch.autograd.grad(v_y, x_req, grad_outputs=ones, create_graph=True)[0][:,1:2]
    r1 = u_t + b1*(u*u_x + v*u_y) + p_x - b2*(u_xx + u_yy)
    r2 = v_t + b1*(u*v_x + v*v_y) + p_y - b2*(v_xx + v_yy)
    r3 = u_x + v_y
    return torch.cat([r1, r2, r3], dim=1)

# Loss combining data and PDE

def loss_fn(model, X_data, uv_data, X_pde, lambda_pde=1.0):
    mse = nn.MSELoss()
    # Data loss (for u, v)
    out_data = model(X_data)
    u_pred = out_data[:,0:1]
    v_pred = out_data[:,1:2]
    loss_data = mse(u_pred, uv_data[:,0:1]) + mse(v_pred, uv_data[:,1:2])
    
    # PDE residual loss
    r_pde = ns_residual(model, X_pde)
    loss_pde = mse(r_pde, torch.zeros_like(r_pde))
    
    total_loss = loss_data + lambda_pde * loss_pde
    return total_loss, loss_data, loss_pde

# Training loop

def train_model(model, data: dict, cfg: dict, device: torch.device):
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'], betas=(0.9,0.999))
    Xd, uvd = data['x_obs'].to(device), data['u_obs'].to(device)
    Xp = data['x_pde'].to(device)
    for ep in range(1, cfg['training']['epochs'] + 1):
        optimizer.zero_grad()
        loss, loss_data, loss_pde = loss_fn(model, Xd, uvd, Xp, cfg['training']['lambda_pde'])
        loss.backward()
        optimizer.step()
        if ep % 2000 == 0:
            print(f"Epoch {ep}: total_loss={loss.item():.4e}, data_loss={loss_data.item():.4e}, pde_loss={loss_pde.item():.4e}")
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, 'ns_inv_model.pth')
    print("Saved initial NS_INV model to 'ns_inv_model.pth'")

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    seed = cfg.get('seed', 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = generate_data(cfg)
    model = InverseNSNet(**cfg['model']).to(device)
    model.train()
    train_model(model, data, cfg, device)


# Entry point
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))  # P_PINN/inverse/nsinv
    project_root = os.path.dirname(os.path.dirname(current_dir))  # P_PINN
    cfg_path = os.path.join(project_root, "configs", "NSInv.yaml")
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str,
                   default=cfg_path,
                   help="path to NSInv.yaml")
    p.add_argument('--seed', type=int, 
                   default=42,
                   help="random seed for reproducibility")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    with open(args.config, 'w') as f:
        yaml.dump(cfg, f)
    main(args.config)