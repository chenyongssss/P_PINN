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

# from common.partition_data import partition_data
from common.pruning import selective_pruning_multi_layers
# change the pruning strategies or pruning criteria
from script.single_step import selective_pruning_multi_layers_single_step
from script.prune_criteria import selective_pruning_multi_layers_abs
from script.prune_criteria import selective_pruning_multi_layers_freq
from script.prune_criteria import selective_pruning_multi_layers_rms
from script.prune_criteria import selective_pruning_multi_layers_std

from common.evaluate import l2_relative_error, mse
from .generate_data import generate_data, true_a, true_u
from .train import pde_residual, loss_fn
from common.pinn_model import PINN


def set_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def fine_tune(model,
              X_good, u_good,
              X_pde,
              X_b_a, a_b,
              X_b_u, u_b,
              X_i_u, u_i,
              epochs=2000, lr=1e-3,
              lambda_pde=1.0, lambda_b=1.0, lambda_i=1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999))
    for epoch in range(epochs):
        optimizer.zero_grad()
        (L, L_data, L_pde, L_a_bdy, L_u_bdy, L_u_ini
         )= loss_fn(model,
                    X_good, u_good,
                    X_pde,
                    X_b_a, a_b,
                    X_b_u, u_b,
                    X_i_u, u_i,
                    lambda_pde, lambda_b, lambda_i)
        L.backward()
        optimizer.step()
        if epoch % 500==0:
            print(f"Fine-tune epoch {epoch}: total={L.item():.4e}, data={L_data.item():.4e}, "
                  f"pde={L_pde.item():.4e}, a_bdy={L_a_bdy.item():.4e}, "
                  f"u_bdy={L_u_bdy.item():.4e}, u_ini={L_u_ini.item():.4e}")
    return model


def evaluate_model(model, nx=60, ny=60, nt=50):
    """
    Evaluate on a finer grid (x,y,t) in [-1,1]^2 x [0,1], returning:
      - L2 relative error (L2RE)
      - L1 relative error (L1RE)
      - MSE
      - MAX error
    for both u and a.
    a is only a function of (x,y) but the network outputs shape with nt dimension.
    """
    x_vals = np.linspace(-1, 1, nx)
    y_vals = np.linspace(-1, 1, ny)
    t_vals = np.linspace(0, 1, nt)
    xx, yy, tt = np.meshgrid(x_vals, y_vals, t_vals, indexing='ij')
    X_test = np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=-1)
    device = next(model.parameters()).device
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(X_test_t)
    u_pred = out[:, 0].cpu().numpy().reshape(nx, ny, nt)
    a_pred = out[:, 1].cpu().numpy().reshape(nx, ny, nt)  # a is repeated in time dimension

    u_true_vals = true_u(xx.ravel(), yy.ravel(), tt.ravel()).reshape(nx, ny, nt)
    a2d = true_a(xx[:, :, 0].ravel(), yy[:, :, 0].ravel()).reshape(nx, ny)
    a3d = np.repeat(a2d[..., None], nt, axis=-1)
    return u_pred, a_pred, u_true_vals, a3d

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
    u_pred = u_pred[:,:1]

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
# Entrypoint
def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    seed = cfg.get('seed',42); set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = generate_data(cfg)
    Xd, ud = data['x_obs'].to(device), data['u_obs'].to(device)
    Xp     = data['X_pde'].to(device)
    Xba, ab= data['X_b_a'].to(device), data['a_b'].to(device)
    Xbu, ub= data['X_b_u'].to(device), data['u_b'].to(device)
    Xiu, ui= data['X_i_u'].to(device), data['u_i'].to(device)

    # load pretrained
    model = PINN(**cfg['model']).to(device)
    ck = torch.load('h_inv_model.pth', map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    u_pred, a_pred, u_true_vals, a3d = evaluate_model(model)
    err_a  = mse(torch.tensor(a_pred), torch.tensor(a3d))
    print(f"Pre-finetune u L2RE={err_a:.4e}")
    model.train()

    # partition
    retain, forget = partition_data(
        Xd, ud, model,
        pde_residual_fn=pde_residual,
        eps=cfg['partition']['eps'],
        w_data=cfg['partition']['w_data'], w_res=cfg['partition']['w_res']
    )

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
        
    Xg, ug = retain
    model  = fine_tune(model, Xg, ug, Xp, Xba, ab,  Xbu, ub, Xiu, ui, cfg['fine_tune']['epochs'],cfg['fine_tune']['lr'],
                       cfg['fine_tune']['lambda_pde'], cfg['fine_tune']['lambda_bdy'], cfg['fine_tune']['lambda_ini'])
    # evaluate on u only
    model.eval()
    u_pred, a_pred, u_true_vals, a3d = evaluate_model(model)
    err_a  = mse(torch.tensor(a_pred), torch.tensor(a3d))
    print(f"Post-finetune u L2RE={err_a:.4e}")

    torch.save({'model_state_dict':model.state_dict(),'config':cfg},'h_inv_finetuned.pth')
    print("Saved finetuned Hinv model to 'h_inv_finetuned.pth'")

if __name__=='__main__':
    cfg_default = os.path.join(root, "P_PINN","configs", "HInv.yaml")
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=cfg_default)
    p.add_argument('--seed',   type=int, default=42)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    with open(args.config, 'w') as f:
        yaml.dump(cfg, f)
    main(args.config)
    