import argparse
import os
import sys
import random
import yaml
import numpy as np
import torch
import torch.optim as optim

# allow imports
current = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(os.path.dirname(current)))
sys.path.append(root)
from common.partition_data import partition_data
from common.pruning import selective_pruning_multi_layers
from common.evaluate import l2_relative_error
from common.pinn_model import PINN
from .generate_data import generate_data
from .train import set_seed, wave_residual, loss_fn



def fine_tune(model, X_good, u_good, X_pde, X_b, u_b, X_i, u_i, epochs=2000, lr=1e-3, 
                   lambda_r=1.0, lambda_b=1.0, lambda_i1=1.0, lambda_i2=1.0, lambda_d=1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    for epoch in range(epochs):
        optimizer.zero_grad()
        L, L_d, L_r, L_b, L_i1, L_i2 = loss_fn(model, X_good, u_good, X_pde, X_b, u_b, X_i, u_i,
                                                      lambda_r, lambda_b, lambda_i1, lambda_i2, lambda_d)
        L.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"Fine-tune epoch {epoch}: total={L.item():.4e}, data={L_d.item():.4e}, pde={L_r.item():.4e}, "
                  f"boundary={L_b.item():.4e}, init={L_i1.item():.4e}, init_t={L_i2.item():.4e}")
    return model

def main(config_path: str):
    # load config and seed
    cfg = yaml.safe_load(open(config_path))
    seed=cfg.get('seed',0); set_seed(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    data=generate_data(cfg)
    Xd, ud = data['X_data'].to(device), data['u_data'].to(device)
    Xp = data['X_pde'].to(device)
    Xb, ub = data['X_b'].to(device), data['u_b'].to(device)
    Xi, ui = data['X_i'].to(device), data['u_i'].to(device)
    Xt, ut = data['X_i'].to(device), data['u_i'].to(device)

    # load model
    md=cfg['model']
    model=PINN(in_dim=2,hidden_dim=md['hidden_dim'],hidden_layers=md['hidden_layers'],out_dim=1,pde_params={'c':md.get('init_c',1.0)}).to(device)
    ck=torch.load('wave_inv_model.pth',map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model.train()

    # partition
    retain, forget=partition_data(
        Xd, ud, model,
        pde_residual_fn=wave_residual,
        eps=cfg['partition']['eps'],
        w_data=cfg['partition']['w_data'], w_res=cfg['partition']['w_res']
    )

    # prune and fine-tune in one main
    # selective pruning
    model=selective_pruning_multi_layers(model,cfg['pruning']['layers'],retain,forget,
                                         alpha=cfg['pruning']['alpha'],num_iter=cfg['pruning']['num_iter'])
    model = fine_tune(model, Xd, ud, Xp, Xb, ub, Xi, ui, cfg['fine_tune']['epochs'], cfg['fine_tune']['lr'],
                cfg['fine_tune']['lambda_pde'], cfg['fine_tune']['lambda_b'], cfg['fine_tune']['lambda_i1'], 
                cfg['fine_tune']['lambda_i2'], cfg['fine_tune']['lambda_d'])
    

    # evaluate
    model.eval()
    with torch.no_grad(): u_pred=model(data['X_test'].to(device))
    rel_u=l2_relative_error(u_pred,data['u_test'].to(device))
    c_err=abs(model.pde_params['c'].item()-2.0)
    print(f"Post-finetune: u L2RE={rel_u:.4e}, c err={c_err:.4e}")

    # save
    torch.save({'model_state_dict':model.state_dict(),'config':cfg},'wave_inv_finetuned.pth')
    print("Saved wave_inv_finetuned.pth")

if __name__=='__main__':
    cfg_default = os.path.join(root, "P_PINN","configs", "WInv.yaml")
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=cfg_default)
    p.add_argument('--seed',   type=int, default=42)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    with open(args.config, 'w') as f:
        yaml.dump(cfg, f)
    main(args.config)