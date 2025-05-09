"""
generate_data.py

Generate synthetic data for 2D Stokes data‐assimilation:

    -Δu + ∇p = 0,   ∇·u = 0   on D = (0,1)^2

Analytic solution:
    u1 = 4·x·y^3,  u2 = x^4 − y^4,  p = 12·x^2·y − 4·y^3 − 1  :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
"""
import numpy as np
import torch

# 1. Exact solution and source term (Stokes equation)
########################################################
def exact_solution_stokes(x):
    """
    Exact solution:
      Velocity: u(x1,x2) = (4*x1*x2^3, x1^4 - x2^4)
      Pressure: p(x1,x2) = 12*x1^2*x2 - 4*x2^3 - 1
    Input x: (N,2) tensor; returns (N,3) tensor with the first two components for velocity and the third for pressure.
    """
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    u1 = 4 * x1 * (x2**3)
    u2 = x1**4 - x2**4
    p  = 12 * (x1**2) * x2 - 4 * (x2**3) - 1
    return torch.cat([u1, u2, p], dim=1)

def source_term_stokes(x):
    """
    Source term is set to zero: For the Stokes equations, we have f = 0 (momentum) and f_d = 0 (divergence constraint).
    Returns (f, f_d) as (N,2) and (N,1) tensors respectively.
    """
    f = torch.zeros((x.shape[0], 2), device=x.device)
    f_d = torch.zeros((x.shape[0], 1), device=x.device)
    return f, f_d

########################################################
# 2. Data generation functions (Stokes equation)
########################################################
def generate_test_data_stokes(n_side=200):
    """
    Construct a uniform grid with n_side x n_side points in D = (0,1)^2 for testing.
    """
    x = np.linspace(0, 1, n_side)
    y = np.linspace(0, 1, n_side)
    XX, YY = np.meshgrid(x, y, indexing='ij')
    X_test = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    u_test = exact_solution_stokes(torch.tensor(X_test, dtype=torch.float32))
    return torch.tensor(X_test, dtype=torch.float32), u_test

def generate_data_points_cartesian_stokes(n_points_per_dim, center, radius):
    """
    Construct Cartesian grid sampling points in subdomain D', which is the circle centered at (0.5, 0.5) with radius 0.25.
    """
    x = np.linspace(0, 1, n_points_per_dim)
    y = np.linspace(0, 1, n_points_per_dim)
    XX, YY = np.meshgrid(x, y, indexing='ij')
    points = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    center = np.array(center)
    
    distances = np.linalg.norm(points - center, axis=1)
    mask = distances <= radius
    data_points = points[mask]
    return torch.tensor(data_points, dtype=torch.float32)

def generate_collocation_points_sobol_stokes(num_points=1280):
    """
    Sample num_points internal points from D = (0,1)^2 using a Sobol sequence (for PDE residual computation).
    """
    sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
    X_int = sobol.draw(num_points)
    return X_int

def add_noise_to_data_stokes(u_clean,noise_noisy, noise_clean, noise_ratio):
    """
    Add Gaussian noise to a fraction (noise_ratio) of the observation data (velocity part).
    """
    N = u_clean.shape[0]
    indices = torch.randperm(N)
    noisy_N = int(noise_ratio * N)
    noisy_idx = indices[:noisy_N]
    clean_idx = indices[noisy_N:]
    u_obs = u_clean.clone()
    # noise = noise_std * torch.randn(u_noisy[noisy_indices].shape)
    # u_noisy[noisy_indices] = u_noisy[noisy_indices] + noise
    u_obs[clean_idx] += noise_clean * torch.randn_like(u_clean[clean_idx])
    u_obs[noisy_idx] += noise_noisy * torch.randn_like(u_clean[noisy_idx])
    return u_obs

def generate_data(cfg: dict) -> dict:
    """
    Builds:
      - x_obs,u_obs: noisy observations inside a circle
      - x_pde: Sobol collocation pts for residual
      - x_test,u_test: full‐domain test grid
    """
    # unpack config
    res        = cfg['data']['obs_circle']['resolution']
    center     = np.array(cfg['data']['obs_circle']['center'])
    radius     = cfg['data']['obs_circle']['radius']
    ratio_clean= cfg['data']['ratio_clean']
    noise_c    = cfg['data']['noise_clean']
    noise_n    = cfg['data']['noise_noisy']
    N_pde      = cfg['data']['pde_points']
    test_n     = cfg['data']['test_grid']


    x_obs = generate_data_points_cartesian_stokes(res, center, radius)
    u_data_full = exact_solution_stokes(x_obs)
    u_data = u_data_full[:, :2]  # only use velocity part
    u_obs = add_noise_to_data_stokes(u_data, noise_n, noise_c, 1-ratio_clean)

    # 1) test grid
    xs = np.linspace(0,1,test_n)
    ys = np.linspace(0,1,test_n)
    XX,YY = np.meshgrid(xs,ys,indexing='ij')
    XYt = np.stack([XX.ravel(), YY.ravel()], axis=1)
    x_test = torch.tensor(XYt, dtype=torch.float32)
    u_test = exact_solution_stokes(x_test)

    

    # 3) collocation (Sobol)
    x_pde =  generate_collocation_points_sobol_stokes(num_points=N_pde)
    return {
        'x_obs':  x_obs,
        'u_obs':  u_obs,
        'x_pde':  x_pde,
        'x_test': x_test,
        'u_test': u_test,
    }
