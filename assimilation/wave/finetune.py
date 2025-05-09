import argparse, yaml, os, sys, random, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from common.pinn_model import PINN
from common.evaluate     import mse
from common.partition_data import partition_data

from common.pruning import selective_pruning_multi_layers
# change the pruning strategies or pruning criteria
from script.single_step import selective_pruning_multi_layers_single_step
from script.prune_criteria import selective_pruning_multi_layers_abs
from script.prune_criteria import selective_pruning_multi_layers_freq
from script.prune_criteria import selective_pruning_multi_layers_rms
from script.prune_criteria import selective_pruning_multi_layers_std


from .generate_data import generate_data
from .generate_data import source_term_wave

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def wave_residual(model, x):
    """
    Compute u_tt -u_xx-f for the wave equation.
    Input x: (N,2) tensor (requires_grad should be True).
    Returns u_tt - u_xx- f, both of shape (N,1).
    """
    u = model(x)
    grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_tt = torch.autograd.grad(u_t, x, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, 1:2]
    f = source_term_wave(x)
    return u_tt-u_xx-f

def fine_tune(model, X_good, u_good, collocation_pts, boundary_pts, lambda_r, lambda_d, lambda_b, adam_epochs, adam_lr, lbfgs_iter):
    """
    Fine-tune the pruned model using the good data from S_d, collocation points, and boundary points.
    First, perform Adam optimization for adam_epochs iterations, then further fine-tune using L-BFGS.
    """
    mse_loss = nn.MSELoss()
    # Phase 1: Adam fine-tuning
    optimizer_adam = optim.Adam(model.parameters(), lr=adam_lr, betas=(0.9, 0.999))
    for epoch in range(adam_epochs):
        optimizer_adam.zero_grad()
        r = wave_residual(model, collocation_pts)
        loss_pde = mse_loss(r, torch.zeros_like(r))
        u_good_pred = model(X_good)
        loss_data = mse_loss(u_good_pred, u_good)
        u_boundary = model(boundary_pts)
        loss_boundary = mse_loss(u_boundary, torch.zeros_like(u_boundary))
        loss = lambda_r * loss_pde + lambda_d * loss_data + lambda_b * loss_boundary
        loss.backward()
        optimizer_adam.step()
        if epoch % 200 == 0:
            print(f"Adam Fine-tune Epoch {epoch}: Total Loss: {loss.item():.6f}, PDE: {loss_pde.item():.6f}, Data: {loss_data.item():.6f}, Boundary: {loss_boundary.item():.6f}")
    # Phase 2: L-BFGS fine-tuning
    optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=lbfgs_iter, history_size=50,
                                   tolerance_grad=1e-9, tolerance_change=1e-9, line_search_fn="strong_wolfe")
    def closure():
        optimizer_lbfgs.zero_grad()
        r = wave_residual(model, collocation_pts)
        
        loss_pde = mse_loss(r, torch.zeros_like(r))
        u_good_pred = model(X_good)
        loss_data = mse_loss(u_good_pred, u_good)
        u_boundary = model(boundary_pts)
        loss_boundary = mse_loss(u_boundary, torch.zeros_like(u_boundary))
        loss = lambda_r * loss_pde + lambda_d * loss_data + lambda_b * loss_boundary
        loss.backward()
        return loss
    optimizer_lbfgs.step(closure)
    print("Fine-tuning (Adam + L-BFGS) on pruned model completed.")
    return model

def main(cfg_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config & set seed
    cfg = yaml.safe_load(open(cfg_path))
    seed = cfg.get('seed', 42)
    set_seed(seed)

    # Load data
    data = generate_data(cfg)
    x_obs = data['x_obs'].to(device).requires_grad_(True)
    u_obs = data['u_obs'].to(device)
    x_pde = data['x_pde'].to(device).requires_grad_(True)
    x_bndry = data['x_bndry'].to(device).requires_grad_(True)
    x_test  = data['x_test'].to(device).requires_grad_(True)
    u_test  = data['u_test'].to(device)

    # Build model & load pruned weights
    model = PINN(**cfg['model']).to(device)
    ckpt = torch.load('wave_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print("Loaded pretrained wave model")

    # Partition obs into good/bad
    retain, forget = partition_data(
        x_obs, u_obs, model,
        pde_residual_fn=wave_residual,   #
        eps=cfg['partition']['eps'],
        w_data=cfg['partition']['w_data'],
        w_res=cfg['partition']['w_res']
    )

    # Selective pruning
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
    X_good, u_good = retain
    model = fine_tune(model, X_good, u_good, x_pde, x_bndry, cfg['fine_tune']['lambda_r'], cfg['fine_tune']['lambda_d'], cfg['fine_tune']['lambda_b'],
                                  cfg['fine_tune']['adam_epochs'], cfg['fine_tune']['lr_adam'],cfg['fine_tune']['adam_epochs'])

    # Final evaluation
    final_mse = mse(model(x_test), u_test)
    print(f"Final MSE on test set: {final_mse:.2e}")

    # Save finetuned model
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_mse': final_mse,
        'config': cfg,
        'seed': seed
    }, 'wave_finetuned_model.pth')
    print("Saved fine‐tuned wave model to 'wave_finetuned_model.pth'")

if __name__ == '__main__':
    cfg_default = os.path.join(project_root, "configs", "Wave.yaml")
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=cfg_default)
    p.add_argument('--seed',   type=int, default=42)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    with open(args.config, 'w') as f:
        yaml.dump(cfg, f)
    main(args.config)
