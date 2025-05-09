"""
generate_data.py

Generate synthetic data for the 1D heat‐equation data‐assimilation problem:

    u_t - u_xx = f,   x ∈ [0,1],  t ∈ [0, T_final]

Analytical solution: u(x,t) = sin(2 pi x) * exp(-4pi^2 t*2).
"""
import numpy as np
import torch


# 2. Exact Solution and Source Term for the Heat Equation
########################################################
def exact_solution_heat(x):
    """
    Exact solution: u(x,t) = exp(-4*pi^2*t^2) * sin(2*pi*x)
    Input x: (N,2) tensor where x[:,0] is the spatial coordinate and x[:,1] is time.
    Returns a (N,1) tensor.
    """
    x_val = x[:, 0:1]
    t_val = x[:, 1:2]
    u = torch.exp(-4*(np.pi**2)*t_val**2) * torch.sin(2*np.pi*x_val)
    return u

def generate_test_data_heat(n_side, T_final):
    # Construct a uniform grid with n_side x n_side points on (0,1) x (0,T)
    x = np.linspace(0, 1, n_side)
    t = np.linspace(0, T_final, n_side)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    X_test = np.stack([xx.ravel(), tt.ravel()], axis=-1)
    u_test = exact_solution_heat(torch.tensor(X_test, dtype=torch.float32))
    return (torch.tensor(X_test, dtype=torch.float32),
            u_test)

def generate_data_points_cartesian_heat(n_x, n_t, T_final):
    """
    Generate a Cartesian grid in the observation domain D' = (0.2, 0.8) x (0, T)
    with a total of n_x x n_t points.
    """
    x = np.linspace(0.2, 0.8, n_x)
    t = np.linspace(0, T_final, n_t)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    X_d = np.stack([xx.ravel(), tt.ravel()], axis=-1)
    return torch.tensor(X_d, dtype=torch.float32)
def add_noise_to_data_heat(data_pts, noise_noisy=0.5, noise_clean=0.01, noise_ratio=1/5):
    """
    Add Gaussian noise to a fraction (noise_ratio) of the observation data.
    """
    u_clean = exact_solution_heat(data_pts)
    N = data_pts.shape[0]
    indices = torch.randperm(N)
    noisy_N = int(noise_ratio * N)
    
    noisy_indices = indices[:noisy_N]
    clean_indices = indices[noisy_N:]
    u_noisy = u_clean.clone()
    
    # 
    noisy_part = noise_noisy * torch.randn(noisy_indices.shape[0], *u_clean.shape[1:])
    clean_part = noise_clean * torch.randn(clean_indices.shape[0], *u_clean.shape[1:])
    
    u_noisy[noisy_indices] += noisy_part
    u_noisy[clean_indices] += clean_part
    return u_noisy

def generate_collocation_points_sobol_heat(num_points, T_final):
    """
    Sample num_points collocation points from the entire domain D = [0,1] x [0,T]
    using a Sobol sequence (for evaluating the PDE residual).
    """
    sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
    X_int = sobol.draw(num_points)
    # First column in [0,1]; scale second column to [0,T]
    X_int[:, 1] = X_int[:, 1] * T_final
    return X_int.requires_grad_(True)

def generate_boundary_points_heat(n_time, T_final):
    """
    Sample data on the boundary ∂D x (0, T) for the 1D spatial domain D = (0, 1).
    The boundary points are at x = 0 and x = 1.
    """
    t = np.linspace(0, T_final, n_time)
    x0 = np.zeros_like(t)
    x1 = np.ones_like(t)
    pts0 = np.stack([x0, t], axis=-1)
    pts1 = np.stack([x1, t], axis=-1)
    pts = np.concatenate([pts0, pts1], axis=0)
    return torch.tensor(pts, dtype=torch.float32)

def generate_data(cfg):
    """
    Create:
      - noisy/clean observations on a regular (x,t) grid,
      - random collocation points for PDE residual,
      - boundary & initial points for BCs/ICs.

    Args:
        cfg (dict): Loaded from Heat.yaml.

    Returns:
        dict with keys:
          x_obs      (Tensor[N_obs,2]),
          u_obs      (Tensor[N_obs,1]),
          x_pde      (Tensor[N_pde,2]),
          x_bndry    (Tensor[N_bndry,2]),
          u_bndry    (Tensor[N_bndry,1])
    """
    # unpack config
    T_final    = cfg['data']['T_final']
    nx, nt     = cfg['data']['obs']['nx'], cfg['data']['obs']['nt']
    noise_noisy = cfg['data']['obs']['noise_noisy']
    noise_clean = cfg['data']['obs']['noise_clean']
    ratio_clean = cfg['data']['obs']['ratio_clean']
    N_pde      = cfg['data']['pde_points']
    N_bndry    = cfg['data']['boundary_points']


    # Generate test data
    x_test, u_test = generate_test_data_heat(n_side=200, T_final=T_final)
    
    # Generate training data:
    x_obs = generate_data_points_cartesian_heat(n_x=nx, n_t=nt, T_final=T_final)   # S_d, 400 points
    u_obs = add_noise_to_data_heat(x_obs, noise_noisy=noise_noisy, noise_clean=noise_clean,noise_ratio=1-ratio_clean)
    
    x_pde = generate_collocation_points_sobol_heat(num_points=N_pde, T_final=T_final)  # S_int, 320 points
    x_bndry = generate_boundary_points_heat(n_time=N_bndry//2, T_final=T_final)  # S_b, 80 points


    
    return {
        'x_obs':   x_obs, 
        'u_obs':   u_obs,
        'x_pde':   x_pde,
        'x_bndry': x_bndry,
        'x_test':  x_test,
        'u_test':  u_test,
    }

