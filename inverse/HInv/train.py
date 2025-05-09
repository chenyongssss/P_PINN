import argparse
import os
import sys
import random
import yaml
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

# allow imports
current = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(os.path.dirname(current)))
sys.path.append(root)

from common.pinn_model import PINN
from .generate_data import generate_data, f_func, true_a, true_u


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def pde_residual(model, x):
    """
    x: [N,3], x[...,0]=x, x[...,1]=y, x[...,2]=t
    Returns r = u_t - div(a * grad(u)) - f.
    """
    x_ = x.clone().detach().requires_grad_(True)
    out = model(x_)
    u  = out[:,0:1]  # shape [N,1]
    a  = out[:,1:2]  # shape [N,1]

    grads_u = torch.autograd.grad(u, x_, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = grads_u[:,2:3]    # partial derivative wrt t
    u_x = grads_u[:,0:1]
    u_y = grads_u[:,1:2]

    a_u_x = a * u_x
    a_u_y = a * u_y

    grads_aux = torch.autograd.grad(a_u_x, x_, grad_outputs=torch.ones_like(a_u_x), create_graph=True)[0]
    grads_auy = torch.autograd.grad(a_u_y, x_, grad_outputs=torch.ones_like(a_u_y), create_graph=True)[0]
    div_val   = grads_aux[:,0:1] + grads_auy[:,1:2]

    # Compute f
    xy = x_.detach().cpu().numpy()
    fx = f_func(xy[:,0], xy[:,1], xy[:,2])
    f_val = torch.tensor(fx, dtype=torch.float32).unsqueeze(-1).to(x_.device)

    return u_t - div_val - f_val

########################################################
# 5. Loss Function
########################################################
def loss_fn(model, 
            X_data, u_data,      # measurement data
            X_pde,              # PDE collocation points
            X_b_a, a_b,         # a boundary
            X_b_u, u_b,         # u spatial boundary
            X_i_u, u_i,         # u initial condition
            lambda_pde=1.0, lambda_b=1.0, lambda_i=1.0):
    """
    total_loss = data_loss + PDE_residual_loss + a_boundary_loss + u_boundary_loss + u_initial_loss
    """
    mse = nn.MSELoss()

    # Data loss
    out_data = model(X_data)
    u_pred_data = out_data[:,0:1]
    loss_data = mse(u_pred_data, u_data)

    # PDE residual loss
    r_pde = pde_residual(model, X_pde)
    loss_pde = mse(r_pde, torch.zeros_like(r_pde))

    # a boundary loss
    out_b_a = model(X_b_a)
    a_pred_b = out_b_a[:,1:2]
    loss_a_bdy = mse(a_pred_b, a_b)

    # u boundary loss
    out_b_u = model(X_b_u)
    u_pred_b = out_b_u[:,0:1]
    loss_u_bdy = mse(u_pred_b, u_b)

    # u initial condition loss
    out_i_u = model(X_i_u)
    u_pred_i = out_i_u[:,0:1]
    loss_u_ini= mse(u_pred_i, u_i)

    total_loss = (loss_data 
                  + lambda_pde*loss_pde 
                  + lambda_b*(loss_a_bdy + loss_u_bdy)
                  + lambda_i*loss_u_ini)
    return total_loss, loss_data, loss_pde, loss_a_bdy, loss_u_bdy, loss_u_ini

def train_model(model, X_data, u_data,
                X_pde,
                X_b_a, a_b,
                X_b_u, u_b,
                X_i_u, u_i,
                epochs=20000, lr=1e-3,
                lambda_pde=1.0, lambda_b=1.0, lambda_i=1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    for epoch in range(epochs):
        optimizer.zero_grad()
        (L, L_data, L_pde, L_a_bdy, L_u_bdy, L_u_ini
         )= loss_fn(model,
                    X_data, u_data,
                    X_pde,
                    X_b_a, a_b,
                    X_b_u, u_b,
                    X_i_u, u_i,
                    lambda_pde, lambda_b, lambda_i)
        L.backward()
        optimizer.step()
        if epoch % 2000 == 0:
            print(f"Epoch {epoch}: total={L.item():.4e}, data={L_data.item():.4e}, "
                  f"pde={L_pde.item():.4e}, a_bdy={L_a_bdy.item():.4e}, "
                  f"u_bdy={L_u_bdy.item():.4e}, u_ini={L_u_ini.item():.4e}")
    return model

# Entrypoint
def main(config_path: str):
    # load config & seed
    cfg = yaml.safe_load(open(config_path))
    seed = cfg.get('seed', 42); set_seed(seed)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    data = generate_data(cfg)
    Xd, ud = data['x_obs'].to(device), data['u_obs'].to(device)
    Xp     = data['X_pde'].to(device)
    Xba, ab= data['X_b_a'].to(device), data['a_b'].to(device)
    Xbu, ub= data['X_b_u'].to(device), data['u_b'].to(device)
    Xiu, ui= data['X_i_u'].to(device), data['u_i'].to(device)

    # model
    model = PINN(**cfg['model']).to(device)
    model.train()

    
    train_model(model, Xd, ud, Xp, Xba, ab, Xbu, ub, Xiu, ui,  cfg['training']['epochs'], cfg['training']['lr'],
                 cfg['training']['lambda_pde'],cfg['training']['lambda_bdy'], cfg['training']['lambda_ini'])

    # save
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, 'h_inv_model.pth')
    print("Saved initial Hinv model to 'h_inv_model.pth'")

if __name__=='__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))  # P_PINN/inverse/nsinv
    project_root = os.path.dirname(os.path.dirname(current_dir))  # P_PINN
    cfg_path = os.path.join(project_root, "configs", "HInv.yaml")
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str,
                   default=cfg_path,
                   help="path to HInv.yaml")
    p.add_argument('--seed', type=int, 
                   default=42,
                   help="random seed for reproducibility")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    with open(args.config, 'w') as f:
        yaml.dump(cfg, f)
    main(args.config)