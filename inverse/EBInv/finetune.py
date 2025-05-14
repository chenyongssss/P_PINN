import argparse
import os
import sys
import random
import yaml
import numpy as np
import torch
import torch.optim as optim

# allow common imports
current = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(os.path.dirname(current)))
sys.path.append(root)

from common.partition_data import partition_data
from common.pruning import selective_pruning_multi_layers

# change the pruning strategies or pruning criteria
from script.single_step import selective_pruning_multi_layers_single_step
from script.prune_criteria import selective_pruning_multi_layers_abs
from script.prune_criteria import selective_pruning_multi_layers_freq
from script.prune_criteria import selective_pruning_multi_layers_rms
from script.prune_criteria import selective_pruning_multi_layers_std

from common.evaluate import l2_relative_error
from common.pinn_model import PINN
from .generate_data import generate_data
from .train import beam_residual, loss_fn

# 1. Set seeds
def set_seed(seed:int=2):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# 2. Fine-tune on retained data
def fine_tune(model, X_good, u_good, X_pde, X_b, u_b, X_i, u_i, epochs=2000, lr=1e-3, 
              lambda_pde=1.0, lambda_b=1.0, lambda_i=1.0, lambda_b_extra=1.0, lambda_i_extra=1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    for epoch in range(epochs):
        optimizer.zero_grad()
        L, L_data, L_pde, L_b, L_b_extra, L_i, L_i_extra = loss_fn(
            model, X_good, u_good, X_pde, X_b, u_b, X_i, u_i,
            lambda_pde, lambda_b, lambda_i, lambda_b_extra, lambda_i_extra)
        L.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"Fine-tune epoch {epoch}: total={L.item():.4e}, data={L_data.item():.4e}, pde={L_pde.item():.4e}, "
                  f"boundary={L_b.item():.4e}, boundary_extra={L_b_extra.item():.4e}, initial={L_i.item():.4e}, initial_extra={L_i_extra.item():.4e}")
    return model

# 3. Evaluation on test set
def evaluate(model, X_test, u_test):
    model.eval()
    with torch.no_grad(): u_pred = model(X_test)
    rel_u = l2_relative_error(u_pred, u_test)
    alpha_err = (model.pde_params['alpha'].item() - 1.0)**2
    print(f"u L2RE={rel_u:.4e}, alpha_err={alpha_err:.4e}")
    return rel_u, alpha_err

def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    seed = cfg.get('seed',42); set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = generate_data(cfg)
    Xd, ud = data['x_obs'].to(device), data['u_obs'].to(device)
    Xp = data['X_pde'].to(device)
    Xb, ub = data['X_b'].to(device), data['u_b'].to(device)
    Xi, ui = data['X_i'].to(device), data['u_i'].to(device)
    Xt, ut = data['X_test'].to(device), data['u_test'].to(device)


    # load pretrained
    md = cfg['model']
    model = PINN(in_dim=2, hidden_dim=md['hidden_dim'], hidden_layers=md['hidden_layers'], out_dim=md['out_dim'], 
                 pde_params={'alpha':md.get('init_alpha',0.0)})
    model.load_state_dict(torch.load('beam_inv_model.pth', map_location=device)['model_state_dict'])
    model.to(device)
    print('Before fine-tune:')
    evaluate(model, Xt, ut)
    
    # partition
    retain, forget = partition_data(
        Xd, ud, model,
        pde_residual_fn=beam_residual,
        eps=cfg['partition']['eps'], w_data=cfg['partition']['w_data'], w_res=cfg['partition']['w_res']
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
    # fine-tune
    Xg, ug = retain
    model = fine_tune(model, Xg, ug,  Xp, Xb, ub, Xi, ui, cfg['fine_tune']['epochs'], cfg['fine_tune']['lr'],
                cfg['fine_tune']['lambda_pde'], cfg['fine_tune']['lambda_b'], cfg['fine_tune']['lambda_i'], 
                cfg['fine_tune']['lambda_b1'], cfg['fine_tune']['lambda_i1'])
    print('After:')
    evaluate(model,Xt,ut)
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, 'beam_inv_finetuned.pth')
    print("Saved beam_inv_finetuned.pth")

# 4. Main
if __name__=='__main__':
    cfg_default = os.path.join(root, "P_PINN","configs", "EInv.yaml")
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=cfg_default)
    p.add_argument('--seed',   type=int, default=42)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    with open(args.config, 'w') as f:
        yaml.dump(cfg, f)
    main(args.config)
    