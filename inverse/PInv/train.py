import argparse
import os
import sys
import random
import yaml
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

# make common modules importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

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

def f_func(x, y):
    """
    According to equation (105):
      f = 2*pi^2*sin(pi*x)*sin(pi*y)/(1+...) 
          + 2*pi*((2*x-1)*cos(pi*x)*sin(pi*y) + (2*y-1)*sin(pi*x)*cos(pi*y))/(1+...)^2
    """
    denom = 1.0 + x**2 + y**2 + (x-1)**2 + (y-1)**2
    part1 = 2 * (np.pi**2) * np.sin(np.pi*x) * np.sin(np.pi*y) / denom
    cosx = np.cos(np.pi*x)
    siny = np.sin(np.pi*y)
    part2 = 2 * np.pi * (((2*x-1)*cosx*siny) + ((2*y-1)*np.sin(np.pi*x)*np.cos(np.pi*y))) / (denom**2)
    fval = part1 + part2
    return fval
def pde_residual(model: PINN, x: torch.Tensor) -> torch.Tensor:
    """
    Compute PDE residual r = -div(a grad u) - f.
    model outputs [u, a].
    """
    x_ = x.clone().detach().requires_grad_(True)
    out = model(x_)
    u = out[:, 0:1]
    a = out[:, 1:2]
    grads_u = torch.autograd.grad(u, x_, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x, u_y = grads_u[:, 0:1], grads_u[:, 1:2]
    a_u_x = a * u_x
    a_u_y = a * u_y
    div_x = torch.autograd.grad(a_u_x, x_, grad_outputs=torch.ones_like(a_u_x), create_graph=True)[0][:, 0:1]
    div_y = torch.autograd.grad(a_u_y, x_, grad_outputs=torch.ones_like(a_u_y), create_graph=True)[0][:, 1:2]
    denom = x_.detach().cpu().numpy()
    f_val = torch.tensor(f_func(denom[:,0], denom[:,1]), dtype=torch.float32, device=x_.device).unsqueeze(-1)
    return -(div_x + div_y) - f_val


def loss_fn(model, X_data, u_data, X_pde, X_b, u_b, a_b,
            lambda_pde=1.0, lambda_b=1.0):
    mse = nn.MSELoss()
    # Data loss: fit u_pred to noisy u_data
    out_data = model(X_data)
    u_pred_data = out_data[:, 0:1]
    loss_data = mse(u_pred_data, u_data)
    # PDE residual loss
    r_pde = pde_residual(model, X_pde)
    loss_pde = mse(r_pde, torch.zeros_like(r_pde))
    # Boundary loss: enforce u=0 and a equals true a_b on the boundary
    out_bdy = model(X_b)
    u_pred_bdy = out_bdy[:, 0:1]
    a_pred_bdy = out_bdy[:, 1:2]
    loss_bdy_u = mse(u_pred_bdy, u_b)
    loss_bdy_a = mse(a_pred_bdy, a_b)
    loss_bdy = loss_bdy_u + loss_bdy_a

    total_loss = loss_data + lambda_pde * loss_pde + lambda_b * loss_bdy
    return total_loss, loss_data, loss_pde, loss_bdy

def train_model(model: PINN, data: dict, cfg: dict, device: torch.device):
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'], betas=(0.9, 0.999))
    epochs = cfg['training']['epochs']
    lambda_pde = cfg['training']['lambda_pde']
    lambda_bdy = cfg['training']['lambda_bdy']
    Xd, ud = data['x_obs'].to(device), data['u_obs'].to(device)
    Xp = data['x_pde'].to(device)
    Xb, ub, ab = data['x_b'].to(device), data['u_b'].to(device), data['a_b'].to(device)
    for ep in range(1, epochs+1):
        optimizer.zero_grad()
        loss, loss_data, loss_pde, loss_bdy = loss_fn(model, Xd, ud, Xp, Xb, ub, ab, lambda_pde, lambda_bdy)
        loss.backward()
        optimizer.step()
        if ep % 500 == 0:
            print(f"Epoch {ep}: total={loss.item():.4e}, data={loss_data.item():.4e}, pde={loss_pde.item():.4e}, bdy={loss_bdy.item():.4e}")
    # save initial model
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, 'poisson_inv_model.pth')
    print("Saved initial Poisson INV model to 'poisson_inv_model.pth'")


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    seed = cfg.get('seed', 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = generate_data(cfg)
    model = PINN(**cfg['model']).to(device)
    model.train()
    train_model(model, data, cfg, device)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))  # P_PINN/inverse/pinv
    project_root = os.path.dirname(os.path.dirname(current_dir))  # P_PINN
    cfg_path = os.path.join(project_root, "configs", "PInv.yaml")
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str,
                   default=cfg_path,
                   help="path to PInv.yaml")
    p.add_argument('--seed', type=int, 
                   default=42,
                   help="random seed for reproducibility")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    with open(args.config, 'w') as f:
        yaml.dump(cfg, f)
    main(args.config)