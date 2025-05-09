import numpy as np
import torch

########################################################
# 1. Exact solution and source term (Wave Equation)
########################################################
def exact_solution_wave(x):
    """
    Exact solution: u(t,x)= sin(2*pi*t)*sin(2*pi*x)
    Input x: (N,2) tensor where x[:,0]=spatial, x[:,1]=time.
    Returns a (N,1) tensor.
    """
    x_val = x[:, 0:1]
    t_val = x[:, 1:2]
    return torch.sin(2 * np.pi * t_val) * torch.sin(2 * np.pi * x_val)

def source_term_wave(x):
    """
    For this analytic solution, the source term f = u_tt - u_xx = 0.
    Returns an (N,1) zero tensor.
    """
    return torch.zeros((x.shape[0], 1), device=x.device)

########################################################
# 2. Test data generation
########################################################
def generate_test_data_wave(n_side=200, T_final=1.0):
    """
    Uniform grid on [0,1]x[0,T_final] with n_side^2 points.
    Returns (X_test, u_test).
    """
    x = np.linspace(0, 1, n_side)
    t = np.linspace(0, T_final, n_side)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    X = np.stack([xx.ravel(), tt.ravel()], axis=-1)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    u_tensor = exact_solution_wave(X_tensor)
    return X_tensor, u_tensor

########################################################
# 3. Observation data in D'_T = ((0,0.2)∪(0.8,1)) × [0,T]
########################################################
def generate_data_points_cartesian_wave(n_x1=10, n_x2=10, n_t=67, T_final=1.0):
    """
    Left subregion x∈[0,0.2], right x∈[0.8,1], both sampled at n_t time steps.
    """
    x1 = np.linspace(0, 0.2, n_x1)
    x2 = np.linspace(0.8, 1, n_x2)
    t  = np.linspace(0, T_final, n_t)
    xx1, tt1 = np.meshgrid(x1, t, indexing='ij')
    xx2, tt2 = np.meshgrid(x2, t, indexing='ij')
    D1 = np.stack([xx1.ravel(), tt1.ravel()], axis=-1)
    D2 = np.stack([xx2.ravel(), tt2.ravel()], axis=-1)
    X_d = np.vstack([D1, D2])
    return torch.tensor(X_d, dtype=torch.float32)

########################################################
# 4. Collocation (Sobol) and Boundary points
########################################################
def generate_collocation_points_sobol_wave(num_points=2160, T_final=1.0):
    """
    Sobol sequence in [0,1]x[0,T_final], for PDE residual.
    """
    sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
    X_int = sobol.draw(num_points)
    X_int[:, 1] *= T_final
    return X_int.requires_grad_(True)

def generate_boundary_points_wave(n_time=50, T_final=1.0):
    """
    Boundary at x=0 and x=1, sampled at n_time points in [0,T_final].
    """
    t = np.linspace(0, T_final, n_time)
    pts0 = np.stack([np.zeros_like(t), t], axis=-1)
    pts1 = np.stack([np.ones_like(t),  t], axis=-1)
    pts = np.vstack([pts0, pts1])
    return torch.tensor(pts, dtype=torch.float32)

########################################################
# 5. Add noise to observations
########################################################
def add_noise_to_data_wave(u_clean, noise_noisy, noise_clean, noise_ratio):
    """
    Corrupt a fraction (noise_ratio) of u_clean with Gaussian noise.
    """
    N = u_clean.shape[0]
    indices = torch.randperm(N)
    n_noise = int(noise_ratio * N)
    noisy_idx, clean_idx = indices[:n_noise], indices[n_noise:]
    u_obs = u_clean.clone()
    u_obs[clean_idx] += noise_clean * torch.randn_like(u_clean[clean_idx])
    u_obs[noisy_idx] += noise_noisy * torch.randn_like(u_clean[noisy_idx])
    return u_obs

########################################################
# 6. Wrapper: generate_data(cfg)
########################################################
def generate_data(cfg):
    """
    Returns a dict with keys:
      x_obs, u_obs, x_pde, x_bndry, x_test, u_test
    using parameters from cfg similar to Heat example.
    """
    T_final    = cfg['data']['T_final']
    n_x1       = cfg['data']['obs']['nx1']
    n_x2       = cfg['data']['obs']['nx2']
    n_t        = cfg['data']['obs']['nt']
    noise_clean  = cfg['data']['obs']['noise_clean']
    noise_noisy  = cfg['data']['obs']['noise_noisy']
    clean_ratio= cfg['data']['obs']['ratio_clean']
    N_pde      = cfg['data']['pde_points']
    N_bndry    = cfg['data']['boundary_points']
    

    # Test data
    n_side = 200
    x_test, u_test = generate_test_data_wave(n_side, T_final)

    # Observations
    x_obs = generate_data_points_cartesian_wave(n_x1, n_x2, n_t, T_final)
    u_obs = add_noise_to_data_wave(exact_solution_wave(x_obs), noise_noisy, noise_clean, 1-clean_ratio)

    # PDE & boundary
    x_pde  = generate_collocation_points_sobol_wave(N_pde, T_final)
    x_bndry = generate_boundary_points_wave(N_bndry//2, T_final)

    # #test data
    # x_test = np.random.rand(10000, 1)
    # t_test = np.random.rand(10000, 1) * T
    # x_test_full = np.concatenate([x_test, t_test], axis=1)
    # x_test_full = torch.tensor(x_test_full, dtype=torch.float32)
    # u_test = exact_solution_wave(x_test_full)

    return {
        'x_obs':   x_obs,
        'u_obs':   u_obs,
        'x_pde':   x_pde,
        'x_bndry': x_bndry,
        'x_test':  x_test,
        'u_test':  u_test,
    }
