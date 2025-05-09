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

from .generate_data import generate_data
from common.pruning import selective_pruning_multi_layers

# change the pruning strategies or pruning criteria
from script.single_step import selective_pruning_multi_layers_single_step
from script.prune_criteria import selective_pruning_multi_layers_abs
from script.prune_criteria import selective_pruning_multi_layers_freq
from script.prune_criteria import selective_pruning_multi_layers_rms
from script.prune_criteria import selective_pruning_multi_layers_std
from common.evaluate import l2_relative_error
from .train import InverseNSNet, loss_fn, ns_residual


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def fine_tune(model, Xg, uvg, Xp, cfg, device):
    optimizer = optim.Adam(model.parameters(), lr=cfg['fine_tune']['lr'],betas=(0.9, 0.999))
    mse = nn.MSELoss()
    for ep in range(1, cfg['fine_tune']['epochs'] + 1):
        optimizer.zero_grad()
        L, L_data, L_pde = loss_fn(model, Xg, uvg, Xp, cfg['fine_tune']['lambda_pde'])
        L.backward()
        optimizer.step()
        if ep % 500 == 0:
            print(f"Fine-tune epoch {ep}: total_loss={L.item():.4e}, data_loss={L_data.item():.4e}, pde_loss={L_pde.item():.4e}")
    
    
    return model

def partition_data(X_data, uv_data, model, eps=0.01, w_data=0.5, w_res=0.5):
    """
    Partition data using a composite error defined as:
       composite_error = w_data * ((u_pred - u_true)^2 + (v_pred - v_true)^2)/2 
                         + w_res * (r1^2 + r2^2 + r3^2)
    Samples with composite_error < eps are labeled "good" (retain).
    """

    model.eval()
    r = ns_residual(model, X_data)
    res_err = torch.sum(r**2, dim=1, keepdim=True)
    with torch.no_grad():
        # 
        X_data = X_data.clone().detach()
        out = model(X_data)
        u_pred = out[:, 0:1]
        v_pred = out[:, 1:2]
        data_err = ((u_pred - uv_data[:, 0:1])**2 + (v_pred - uv_data[:, 1:2])**2) / 2.0
        
        # 
    

    # 
    composite_err = (w_data * data_err + w_res * res_err).squeeze().cpu().detach().numpy()
    good_idx = np.where(composite_err < eps)[0]
    bad_idx = np.where(composite_err >= eps)[0]
    print(f"Composite error threshold eps={eps:.6e}")
    print(f"Good points = {len(good_idx)}, Bad points = {len(bad_idx)}")
    
    X_good = X_data[good_idx]
    uv_good = uv_data[good_idx]
    X_bad = X_data[bad_idx]
    uv_bad = uv_data[bad_idx]
    return (X_good, uv_good), (X_bad, uv_bad)

# Evaluation helper

def evaluate(m, data, device):
    X_test, uv_test, p_test = data['x_test'].to(device), data['uv_test'].to(device), data['p_test'].to(device)
    m.eval()
    with torch.no_grad():
        out = m(X_test)
    u_err = l2_relative_error(out[:,0:1], uv_test[:,0:1])
    v_err = l2_relative_error(out[:,1:2], uv_test[:,1:2])
    p_err = l2_relative_error(out[:,2:3], p_test)
    print(f"Test errors: u L2RE={u_err:.4e}, v L2RE={v_err:.4e}, p L2RE={p_err:.4e}")
    return u_err, v_err, p_err

def main(cfg_path: str):

    # load config and seed
    cfg = yaml.safe_load(open(cfg_path))
    seed = cfg.get('seed', 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = generate_data(cfg)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    Xd, ud = data['x_obs'].to(device).requires_grad_(True), data['u_obs'].to(device)
    Xp = data['x_pde'].to(device).requires_grad_(True)
    # Xt, uvt, pt = data['x_test'].to(device), data['uv_test'].to(device), data['p_test'].to(device)

    model = InverseNSNet(**cfg['model']).to(device)
    model.load_state_dict(torch.load('ns_inv_model.pth', map_location=device)['model_state_dict'])
    evaluate(model, data, device)
    retain, forget = partition_data(Xd, ud, model,
                                   cfg['partition']['eps'], cfg['partition']['w_data'], cfg['partition']['w_res'])
    
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

    x_good, u_good = retain
    model = fine_tune(model, x_good, u_good, Xp, cfg, device)
    evaluate(model, data, device)
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, 'ns_inv_finetuned.pth')
    print("Saved finetuned NS_INV model to 'ns_inv_finetuned.pth'")
if __name__ == '__main__':
    cfg_default = os.path.join(root, "P_PINN","configs", "NSInv.yaml")
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=cfg_default)
    p.add_argument('--seed',   type=int, default=42)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    with open(args.config, 'w') as f:
        yaml.dump(cfg, f)
    main(args.config)
    