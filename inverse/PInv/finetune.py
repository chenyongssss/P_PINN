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
from common.partition_data import partition_data
from common.pruning import selective_pruning_multi_layers

# change the pruning strategies or pruning criteria
from script.single_step import selective_pruning_multi_layers_single_step
from script.prune_criteria import selective_pruning_multi_layers_abs
from script.prune_criteria import selective_pruning_multi_layers_freq
from script.prune_criteria import selective_pruning_multi_layers_rms
from script.prune_criteria import selective_pruning_multi_layers_std
from common.evaluate import l2_relative_error
from .generate_data import generate_data
from .train import pde_residual, loss_fn

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def fine_tune(model, X_good, u_good, X_pde, X_b, u_b, a_b, epochs=2000, lr=1e-3, lambda_pde=1.0, lambda_b=1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    for epoch in range(epochs):
        optimizer.zero_grad()
        L, L_data, L_pde, L_bdy = loss_fn(model, X_good, u_good, X_pde, X_b, u_b, a_b, lambda_pde, lambda_b)
        L.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Fine-tune epoch {epoch}: total={L.item():.4e}, data={L_data.item():.4e}, pde={L_pde.item():.4e}, bdy={L_bdy.item():.4e}")
    return model

def main(cfg_path: str):
    # load config and seed
    cfg = yaml.safe_load(open(cfg_path))
    seed = cfg.get('seed', 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = generate_data(cfg)
    Xd, ud = data['x_obs'].to(device), data['u_obs'].to(device)
    Xp = data['x_pde'].to(device)
    Xb, ub, ab = data['x_b'].to(device), data['u_b'].to(device), data['a_b'].to(device)
    Xt, ut, at = data['x_test'].to(device), data['u_test'].to(device), data['a_test'].to(device)

    # build & load pretrained model
    model = PINN(**cfg['model']).to(device)
    ckpt = torch.load('poisson_inv_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
     # evaluate on test
    model.eval()
    with torch.no_grad():
        pred = model(Xt)
        err_u = l2_relative_error(pred[:,0:1], ut)
        err_a = l2_relative_error(pred[:,1:2], at)
    print(f"Test Rel-L2 error u: {err_u:.4e}, a: {err_a:.4e}")
    model.train()

    # partition data
    retain, forget = partition_data(
        Xd, ud, model,
        pde_residual_fn=pde_residual,
        eps=cfg['partition']['threshold'],
        w_data=cfg['partition']['w_data'],
        w_res=cfg['partition']['w_res']
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
    
    # model = selective_pruning_multi_layers(
    #     model,
    #     layer_indices=cfg['pruning']['layers'],
    #     retain_data=retain,
    #     forget_data=forget,
    #     alpha=cfg['pruning']['alpha'],
    #     num_iter=cfg['pruning']['num_iter']
    # )

   
    Xg, ug = retain
    model = fine_tune(model, Xg, ug, Xp, Xb, ub, ab, cfg['finetune']['epochs'], cfg['finetune']['lr'], cfg['finetune']['lambda_pde'], 
                      cfg['finetune']['lambda_bdy'])
    # evaluate on test
    model.eval()
    with torch.no_grad():
        pred = model(Xt)
        err_u = l2_relative_error(pred[:,0:1], ut)
        err_a = l2_relative_error(pred[:,1:2], at)
    print(f"Test Rel-L2 error u: {err_u:.4e}, a: {err_a:.4e}")

    # save finetuned model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg
    }, 'poisson_inv_finetuned.pth')
    print("Saved finetuned model to 'poisson_inv_finetuned.pth'")


if __name__ == '__main__':
    cfg_default = os.path.join(project_root, "P_PINN","configs", "PInv.yaml")
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=cfg_default)
    p.add_argument('--seed',   type=int, default=42)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    with open(args.config, 'w') as f:
        yaml.dump(cfg, f)
    main(args.config)