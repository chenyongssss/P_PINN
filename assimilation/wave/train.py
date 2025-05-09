import argparse, yaml, os, sys, random, numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

# Allow imports of common & local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from common.pinn_model import PINN
from common.evaluate     import mse
from common.partition_data import partition_data
from common.pruning      import selective_pruning_multi_layers

# Wave‐specific utilities
from .generate_data             import generate_data
from .generate_data             import source_term_wave, exact_solution_wave
from .generate_data             import generate_collocation_points_sobol_wave
from .generate_data             import generate_boundary_points_wave


def compute_wave_residual(model, x):
    """
    Compute u_tt and u_xx for the wave equation.
    Input x: (N,2) tensor (requires_grad should be True).
    Returns u_tt and u_xx, both of shape (N,1).
    """
    u = model(x)
    grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_tt = torch.autograd.grad(u_t, x, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, 1:2]
    return u_tt, u_xx
def pretrain_data(model, data_pts, u_data, epochs, lr):
    """
    Pretrain the model using only the data loss with the Adam optimizer.
    """
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999))
    for epoch in range(epochs):
        optimizer.zero_grad()
        u_pred = model(data_pts)
        loss = mse_loss(u_pred, u_data)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Pretrain Epoch {epoch}, Data Loss: {loss.item():.6f}")
    print("Pretraining (data loss only) completed.")
    return model

def compute_adaptive_weights_all(model, collocation_pts, data_pts, u_data, boundary_pts, eps=1e-3, lambda_d_max=1e3, lambda_b_max=1e3):
    """
    Compute adaptive weights for the composite loss (PDE + data + boundary).
    """
    mse_loss = nn.MSELoss()
    # PDE loss
    u_tt, u_xx = compute_wave_residual(model, collocation_pts)
    f = source_term_wave(collocation_pts)
    loss_pde = mse_loss(u_tt - u_xx - f, torch.zeros_like(f))
    grads_r = torch.autograd.grad(loss_pde, model.parameters(), retain_graph=True, allow_unused=True)
    trace_r = sum(torch.sum(g**2) if g is not None else 0.0 for g in grads_r)
    N_r = collocation_pts.shape[0]
    
    # Data loss
    u_data_pred = model(data_pts)
    loss_data = mse_loss(u_data_pred, u_data)
    grads_d = torch.autograd.grad(loss_data, model.parameters(), retain_graph=True, allow_unused=True)
    trace_d = sum(torch.sum(g**2) if g is not None else 0.0 for g in grads_d)
    N_d = data_pts.shape[0]
    
    # Boundary loss
    u_boundary = model(boundary_pts)
    loss_boundary = mse_loss(u_boundary, torch.zeros_like(u_boundary))
    grads_b = torch.autograd.grad(loss_boundary, model.parameters(), retain_graph=True, allow_unused=True)
    trace_b = sum(torch.sum(g**2) if g is not None else 0.0 for g in grads_b)
    N_b = boundary_pts.shape[0]
    
    R = (trace_r / N_r) + (trace_d / N_d) + (trace_b / N_b)
    lambda_r = (N_r * R) / (trace_r + eps)
    lambda_d = (N_d * R) / (trace_d + eps)
    lambda_b = (N_b * R) / (trace_b + eps)
    
    if lambda_d > lambda_d_max:
        lambda_d = torch.tensor(lambda_d_max, device=collocation_pts.device)
    if lambda_b > lambda_b_max:
        lambda_b = torch.tensor(lambda_b_max, device=collocation_pts.device)
    
    print(f"Adaptive weights computed: lambda_r={lambda_r.item():.6e}, lambda_d={lambda_d.item():.6e}, lambda_b={lambda_b.item():.6e}")
    return lambda_r, lambda_d, lambda_b

def fine_tune_composite_lbfgs(model, collocation_pts, data_pts, u_data, boundary_pts, lambda_r, lambda_d, lambda_b, max_iter):
    """
    Fine-tune the model using composite loss (PDE + data + boundary) with L-BFGS.
    """
    mse_loss = nn.MSELoss()
    optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=max_iter, history_size=50,
                                  tolerance_grad=1e-9, tolerance_change=1e-9, line_search_fn="strong_wolfe")
    
    def closure():
        optimizer_lbfgs.zero_grad()
        # PDE loss
        u_tt, u_xx = compute_wave_residual(model, collocation_pts)
        f = source_term_wave(collocation_pts)
        loss_pde = mse_loss(u_tt - u_xx - f, torch.zeros_like(f))
        # Data loss
        u_data_pred = model(data_pts)
        loss_data = mse_loss(u_data_pred, u_data)
        # Boundary loss
        u_boundary = model(boundary_pts)
        loss_boundary = mse_loss(u_boundary, torch.zeros_like(u_boundary))
        loss = lambda_r * loss_pde + lambda_d * loss_data + lambda_b * loss_boundary
        loss.backward()
        return loss
    
    optimizer_lbfgs.step(closure)
    print("Composite fine-tuning with L-BFGS completed.")
    return model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(cfg_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    cfg = yaml.safe_load(open(cfg_path))
    seed = cfg.get('seed', 42)
    set_seed(seed)

    # Generate data
    data = generate_data(cfg)
    x_obs   = data['x_obs'].to(device).requires_grad_(True)
    u_obs   = data['u_obs'].to(device)
    x_pde   = data['x_pde'].to(device).requires_grad_(True)
    x_bndry =  data['x_bndry'].to(device).requires_grad_(True)
    # x_test = data['x_test'].to(device)
    # u_test = data['u_test'].to(device)

    # Build model
    model = PINN(**cfg['model']).to(device)
    model.train()

    # 1. Adam pretraining (data only)
    pretrain_data(model, x_obs, u_obs,
                  epochs=cfg['training']['pretrain_epochs'],
                  lr=cfg['training']['lr_pretrain'])

    # 2. Compute adaptive weights
    lambda_r, lambda_d, lambda_b = compute_adaptive_weights_all(
        model, x_pde, x_obs, u_obs,
        boundary_pts=x_bndry,
        eps=cfg['training']['adaptive_eps'],
        lambda_d_max=cfg['training']['lambda_d_max'],
        lambda_b_max=cfg['training']['lambda_b_max']
    )

    # 3. L-BFGS composite fine‐tuning
    fine_tune_composite_lbfgs(
        model,
        collocation_pts=x_pde,
        data_pts=x_obs,
        u_data=u_obs,
        boundary_pts=x_bndry,
        lambda_r=lambda_r,
        lambda_d=lambda_d,
        lambda_b=lambda_b,
        max_iter=cfg['training']['lbfgs_max_iter']
    )

    # Save pretrained wave model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'seed': seed,
        'device': str(device)
    }, 'wave_model.pth')
    print("Pre-trained & refined wave model saved to 'wave_model.pth'")

if __name__ == '__main__':
    cfg_default = os.path.join(project_root, "configs", "Wave.yaml")
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=cfg_default,
                   help="path to Wave.yaml")
    p.add_argument('--seed',   type=int, default=42,
                   help="random seed")
    args = p.parse_args()
    # write back seed into config
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    with open(args.config, 'w') as f:
        yaml.dump(cfg, f)
    main(args.config)
