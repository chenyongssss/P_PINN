import numpy as np
import torch

def generate_grid_data(nx: int, nt: int):
    """
    Generate full grid data for Euler–Bernoulli beam:
      u(x,t) = sin(pi*x)*cos(alpha*pi^2*t) with true alpha=1.0.
    Returns X_all (N,2) and u_all (N,1) numpy arrays.
    """
    true_alpha = 1.0
    x_vals = np.linspace(0,1,nx)
    t_vals = np.linspace(0,1,nt)
    xx, tt = np.meshgrid(x_vals, t_vals, indexing='ij')
    X_all = np.stack([xx.ravel(), tt.ravel()], axis=-1)
    u_all = (np.sin(np.pi*xx) * np.cos(true_alpha*(np.pi**2)*tt)).ravel()[:,None]
    return X_all, u_all


def sample_collocation_points(num_pde: int):
    """
    Uniformly sample collocation points in domain [0,1]^2 for PDE residual.
    """
    X_pde = np.random.rand(num_pde, 2)
    return X_pde


def sample_initial_points(num_ini: int):
    """
    Sample initial condition points (t=0): u(x,0)=sin(pi*x).
    """
    x = np.linspace(0,1,num_ini)[:,None]
    t = np.zeros_like(x)
    X_i = np.hstack([x,t])
    u_i = np.sin(np.pi*x)
    return X_i, u_i


def sample_boundary_points(num_bdy: int):
    """
    Sample boundary condition points at x=0 or x=1: u=0.
    """
    pts, vals = [], []
    for _ in range(num_bdy):
        side = np.random.choice([0,1])
        x_val = float(side)
        t_val = np.random.rand()
        pts.append([x_val, t_val])
        vals.append([0.0])
    return np.array(pts), np.array(vals)

def create_mixed_labeled_data(x_d, u_d, noise_noisy=1.0, noise_clean=0.0, noise_ratio=2/5):
    """
    For each point, randomly choose between clean labels (noise=0.01) and noisy labels (noise=1.0) in a 2:3 ratio.
    """
   
    
    N = u_d.shape[0]
    noisy_N = int(noise_ratio * N)
    # Shuffle indices
    indices = torch.randperm(N)
    noisy_idx = indices[:noisy_N]
    clean_idx = indices[noisy_N:]
    u_obs = torch.tensor(u_d, dtype=torch.float32)
    
    # 
    u_obs[clean_idx] += noise_clean * torch.randn_like(u_obs[clean_idx])
    u_obs[noisy_idx] += noise_noisy * torch.randn_like(u_obs[noisy_idx])
    return torch.tensor(x_d, dtype=torch.float32), u_obs

def generate_data(cfg: dict) -> dict:
    """
    Generate and return all datasets according to cfg['data']:
      nx, nt, train_size,
      noisy_ratio, noise_noisy_std, noise_clean_std,
      pde_points, boundary_points, initial_points
    Returns dict with tensors:
      X_data, u_data, X_pde, X_b, u_b, X_i, u_i, X_test, u_test
    """
    d = cfg['data']
    # full grid
    X_all, u_all = generate_grid_data(d['nx'], d['nt'])

     # Randomly sample 10000 points as labeled data, rest as test data
    total_points = X_all.shape[0]
    indices = np.arange(total_points)
    np.random.shuffle(indices)
    train_indices = indices[:5000]
    test_indices  = indices[5000:]
    X_train_np = X_all[train_indices]
    u_train_np = u_all[train_indices]
    X_test_np  = X_all[test_indices]
    u_test_np  = u_all[test_indices]

    x_obs, u_obs = create_mixed_labeled_data(X_train_np, u_train_np, d['noise_noisy'], d['noise_clean'], 1-d['ratio_clean'])
    
    # collocation, boundary, initial
    X_pde = sample_collocation_points(d['pde_points'])
    X_i, u_i = sample_initial_points(d['initial_points'])
    X_b, u_b = sample_boundary_points(d['boundary_points'])
    # convert to torch
    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32)
    return {
        'x_obs': x_obs,
        'u_obs': u_obs,
        'X_pde' : to_tensor(X_pde),
        'X_b'   : to_tensor(X_b),
        'u_b'   : to_tensor(u_b),
        'X_i'   : to_tensor(X_i),
        'u_i'   : to_tensor(u_i),
        'X_test': to_tensor(X_test_np),
        'u_test': to_tensor(u_test_np)
    }