"""
finetune.py

Fine‐tune the pruned Poisson PINN on the retained observations + PDE.
Saves final weights to 'finetuned_model.pth'.
"""
import argparse
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from .generate_data import generate_data, source_term
import random
import numpy as np
import os
import sys


current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(project_root)
from common.pinn_model import PINN
from common.partition_data import partition_data
from common.evaluate import mse
from common.pruning import selective_pruning_multi_layers
# change the pruning strategies or pruning criteria
from script.single_step import selective_pruning_multi_layers_single_step
from script.prune_criteria import selective_pruning_multi_layers_abs
from script.prune_criteria import selective_pruning_multi_layers_freq
from script.prune_criteria import selective_pruning_multi_layers_rms
from script.prune_criteria import selective_pruning_multi_layers_std

# Set seeds for reproducibility
def set_seed(seed=42):
    """
    Set seeds for reproducibility across Python, NumPy and PyTorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")
def compute_laplacian(u, x):
    grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    u_x, u_y = grads[:,0:1], grads[:,1:2]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:,0:1]
    u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y),
                               create_graph=True)[0][:,1:2]
    return u_xx + u_yy

def poisson_residual(model, x):
    u = model(x)
    lap = compute_laplacian(u, x)
    f   = source_term(x)
    return lap + f

def fine_tune(
    model: nn.Module,
    X_good: torch.Tensor,
    u_good: torch.Tensor,
    x_pde: torch.Tensor,
    pde_residual_fn,
    fine_cfg: dict
) -> nn.Module:
    """
    Fine-tune a pruned PINN model in two stages:
      1) Adam optimization on retained data + PDE residual loss
      2) L-BFGS refinement on the same composite loss

    Args:
      model:            The pruned PINN model to fine-tune.
      X_good, u_good:   High‐quality observation inputs and targets.
      x_pde:            Collocation points for PDE residual evaluation.
      pde_residual_fn:  Function(model, x) -> PDE residual tensor.
      fine_cfg:         Dictionary containing fine‐tuning hyperparameters:
                        'lr_adam', 'adam_epochs',
                        'lambda_d', 'lambda_p',
                        'lbfgs_epochs'.
    """
    mse_loss = nn.MSELoss()

    # Stage 1: Adam optimization
    optimizer_adam = optim.Adam(
        model.parameters(),
        lr=fine_cfg['lr_adam'],
        betas=(0.9, 0.999)
    )
    for epoch in range(1, fine_cfg['adam_epochs'] + 1):
        optimizer_adam.zero_grad()
        # Compute data fidelity loss
        pred = model(X_good)
        loss_data = mse_loss(pred, u_good)
        # Compute PDE residual loss
        res = pde_residual_fn(model, x_pde)
        loss_pde = mse_loss(res, torch.zeros_like(res))
        # Composite loss
        loss = fine_cfg['lambda_d'] * loss_data \
             + fine_cfg['lambda_p'] * loss_pde
        loss.backward()
        optimizer_adam.step()
        if epoch % 500 == 0:
            print(f"[Adam FT] Epoch {epoch:4d} | Loss={loss.item():.2e}")

    # Stage 2: L-BFGS refinement
    optimizer_lb = optim.LBFGS(
        model.parameters(),
        max_iter=fine_cfg['lbfgs_epochs'],
        history_size=50,
        tolerance_grad=1e-9,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer_lb.zero_grad()
        pred = model(X_good)
        loss_data = mse_loss(pred, u_good)
        res = pde_residual_fn(model, x_pde)
        loss_pde = mse_loss(res, torch.zeros_like(res))
        loss = fine_cfg['lambda_d'] * loss_data \
             + fine_cfg['lambda_p'] * loss_pde
        loss.backward()
        return loss

    optimizer_lb.step(closure)
    print("L-BFGS fine-tuning complete.")

    return model


def main(cfg_path):
    # Set device: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # load config
    cfg = yaml.safe_load(open(cfg_path))
    
    # Use seed from config if available
    seed = 42  # Default seed
    if 'seed' in cfg:
        seed = cfg['seed']
    
    # Set all seeds
    set_seed(seed)
    
    # load data
    data = generate_data(cfg)
    x_obs = data['x_obs'].to(device).requires_grad_(True)
    u_obs = data['u_obs'].to(device)
    x_pde = data['x_pde'].to(device).requires_grad_(True)
    x_test = data['x_test'].to(device).requires_grad_(True)
    u_test = data['u_test'].to(device)
    
    # rebuild model
    model = PINN(**cfg['model']).to(device)
    
    # Load pruned model weights
    try:
        # Try loading the newer format with metadata
        checkpoint = torch.load('poisson_model.pth', map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded pruned model with metadata")
        else:
            # Fall back to old format (direct state dict)
            model.load_state_dict(checkpoint)
            print("Loaded pruned model (legacy format)")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the pruned model was saved correctly")
        return
    # model.train()
    
    print("baseline-pinn:")
    init_mse = mse(model(x_test), u_test)
    print(f"Init MSE on all test points: {init_mse:.2e}")
    

     # selective pruning
    retain, forget = partition_data(
        x_obs, u_obs, model,
        pde_residual_fn = poisson_residual,
        eps=cfg['partition']['eps'],
        w_data=cfg['partition']['w_data'],
        w_res=cfg['partition']['w_res']
    )
    X_good, u_good = retain
    X_bad,  u_bad  = forget


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


    # Fine-tune the pruned model using the extracted function
    model = fine_tune(
        model,
        X_good, u_good,
        x_pde,
        poisson_residual,
        fine_cfg=cfg['fine_tune']
    )

    # final evaluation
    final_mse = mse(model(x_test), u_test)
    print(f"Final MSE on all test points: {final_mse:.2e}")

    torch.save(model.state_dict(), 'poisson_finetuned_model.pth')
    print("Saved fine‐tuned model to 'poisson_finetuned_model.pth'.")

if __name__=='__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))  # P_PINN/assimilation/poisson
    project_root = os.path.dirname(os.path.dirname(current_dir))  # P_PINN
    cfg_path = os.path.join(project_root, "configs", "Poisson.yaml")
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str,
                   default=cfg_path,
                   help="path to Poisson.yaml")
    p.add_argument('--seed', type=int, 
                   default=42,
                   help="random seed for reproducibility")
    args = p.parse_args()
    
    # Override seed in config if provided through command line
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    with open(args.config, 'w') as f:
        yaml.dump(cfg, f)
        
    main(args.config)
