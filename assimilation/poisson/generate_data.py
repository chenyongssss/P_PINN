"""
generate_data.py

Generate synthetic data for the 2D Poisson data‐assimilation problem:

    Δu(x,y) + f(x,y) = 0,    (x,y) ∈ (0,1)^2

Exact solution: u(x,y) = 30·x·(1−x)·y·(1−y),
Source term:    f(x,y) = 60·[x(1−x) + y(1−y)].
"""
import numpy as np
import torch



def generate_test_data(n_side=200):
    # Construct an n_side x n_side uniform grid over the domain D = (0,1)^2
    x = np.linspace(0, 1, n_side)
    y = np.linspace(0, 1, n_side)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    X_test = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    # Exact solution: u(x,y)=30*x*(1-x)*y*(1-y)
    u_test = 30 * xx * (1 - xx) * yy * (1 - yy)
    return (torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(u_test, dtype=torch.float32))

def generate_data_points_cartesian(n_side=15):
    """
    Construct a Cartesian grid over the observation subdomain D' = [0.125,0.875]^2
    with n_side x n_side points.
    """
    x = np.linspace(0.125, 0.875, n_side)
    y = np.linspace(0.125, 0.875, n_side)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    X_d = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    return torch.tensor(X_d, dtype=torch.float32)

def generate_collocation_points_sobol(num_points=175):
    """
    Use a Sobol sequence to sample num_points collocation points over the domain D = [0,1]^2 (for PDE residual).
    """
    sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
    X_int = sobol.draw(num_points).cpu().numpy()
    return torch.tensor(X_int, dtype=torch.float32, requires_grad=True)

def add_noise_to_data(data_pts, noise_noisy, noise_clean, noise_ratio):
    """
    Add noise to the exact labels corresponding to data_pts:
      - Compute clean labels using exact_solution(data_pts)
      - Randomly select a fraction (noise_ratio) of the data and add Gaussian noise (std=noise_std) to their labels; the rest remain clean.
    Returns the noisy labels u_data.
    """
    u_clean = exact_solution(data_pts)
    N = data_pts.shape[0]
    indices = torch.randperm(N)
    n_noise = int(noise_ratio * N)
    noisy_idx, clean_idx = indices[:n_noise], indices[n_noise:]
    u_obs = u_clean.clone()
    u_obs[clean_idx] += noise_clean * torch.randn_like(u_clean[clean_idx])
    u_obs[noisy_idx] += noise_noisy * torch.randn_like(u_clean[noisy_idx])
    
    return u_obs

########################################################
# 2. Exact Solution and Source Term
########################################################
def exact_solution(x):
    """
    Exact solution: u(x,y)=30*x*(1-x)*y*(1-y)
    Input x: Tensor of shape (N,2); returns a tensor of shape (N,1).
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    u = 30 * x1 * (1 - x1) * x2 * (1 - x2)
    return u.unsqueeze(1)

def source_term(x):
    """
    Source term: f(x)=60*(x1*(1-x1)+x2*(1-x2))
    Corresponds to the PDE: Δu + f = 0.
    Input x: Tensor of shape (N,2); returns a tensor of shape (N,1).
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    f = 60 * (x1 * (1 - x1) + x2 * (1 - x2))
    return f.unsqueeze(1)


def generate_data(cfg: dict) -> dict:
    """
    Create:
      - observations (x_obs,u_obs) in subdomain D' = [0.125,0.875]^2
      - collocation points x_pde in full D = [0,1]^2
      - test grid (x_test,u_test) over full domain

    Args:
        cfg: loaded from Poisson.yaml
    Returns:
        dict with keys 'x_obs','u_obs','x_pde','x_test','u_test'
    """
    # unpack config
    test_grid   = cfg['data']['test_grid']
    obs_grid    = cfg['data']['obs_grid']
    noise_noisy = cfg['data']['noise_noisy']
    noise_clean = cfg['data']['noise_clean']
    ratio_clean = cfg['data']['ratio_clean']
    N_pde       = cfg['data']['pde_points']

    # 1) test grid
    xs = np.linspace(0,1,test_grid)
    ys = np.linspace(0,1,test_grid)
    XX,YY = np.meshgrid(xs,ys,indexing='ij')
    XYt = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    x_test = torch.tensor(XYt, dtype=torch.float32)
    u_test = exact_solution(x_test)

    # 2) obs grid in subdomain
    x_obs = generate_data_points_cartesian(n_side=obs_grid)
   
    u_obs = add_noise_to_data(x_obs, noise_noisy,noise_clean,1-ratio_clean)

    
   

    # 3) collocation (Sobol)
    x_pde = generate_collocation_points_sobol(num_points=N_pde)
    x_pde.requires_grad_(True)

    return {
        'x_obs':  x_obs,
        'u_obs':  u_obs,
        'x_pde':  x_pde,
        'x_test': x_test,
        'u_test': u_test,
    }
