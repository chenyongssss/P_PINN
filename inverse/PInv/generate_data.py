import numpy as np
import torch

# 1. Define true solution and coefficient functions

def true_u(x, y):
    """
    Analytical solution u(x,y) = sin(pi*x) * sin(pi*y).
    """
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def true_a(x, y):
    """
    Coefficient a(x,y) = 1 / (1 + x^2 + y^2 + (x-1)^2 + (y-1)^2).
    """
    return 1.0 / (1.0 + x**2 + y**2 + (x-1)**2 + (y-1)**2)

# 2. Compute source term f for PDE: -div(a grad u) = f

def f_func(x, y):
    """
    Source term derived for the given u and a.
    """
    denom = 1.0 + x**2 + y**2 + (x-1)**2 + (y-1)**2
    part1 = 2 * (np.pi**2) * np.sin(np.pi*x) * np.sin(np.pi*y) / denom
    cosx, siny = np.cos(np.pi*x), np.sin(np.pi*y)
    part2 = 2 * np.pi * (
        ((2*x-1)*cosx*siny) + ((2*y-1)*np.sin(np.pi*x)*np.cos(np.pi*y))
    ) / (denom**2)
    return part1 + part2

# 3. Generate noisy observation data on a grid

def generate_mixed_labeled_data(nx=50, ny=50, noise_noisy=1.0, noise_clean=0.0, noise_ratio=2/5):
    """
    Generate a 50x50 grid on [0,1]^2 with 2500 points.
    For each point, randomly choose between clean labels (noise=0.01) and noisy labels (noise=1.0) in a 2:3 ratio.
    """
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x_vals, y_vals)
    X = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # shape: [2500, 2]
    u_true_vals = true_u(xx.ravel(), yy.ravel())
    u_true_vals = torch.tensor(u_true_vals.reshape(-1, 1), dtype=torch.float32)
    total_points = nx * ny
    noisy_N = int(noise_ratio * total_points)
    # Shuffle indices
    indices = torch.randperm(total_points)
    noisy_idx = indices[:noisy_N]
    clean_idx = indices[noisy_N:]
    u_obs = u_true_vals.clone()
    
    # 
    u_obs[clean_idx] += noise_clean * torch.randn_like(u_true_vals[clean_idx])
    u_obs[noisy_idx] += noise_noisy * torch.randn_like(u_true_vals[noisy_idx])
    return torch.tensor(X, dtype=torch.float32), u_obs

# 4. Generate collocation points for PDE residual

def generate_collocation_points(num_pde=8192):
    """
    Sample num_pde random points in [0,1]^2 for computing PDE residual.
    """
    Xpde = np.random.rand(num_pde, 2)
    return torch.tensor(Xpde, dtype=torch.float32)

# 5. Generate boundary points with Dirichlet conditions

def generate_boundary_points(num_bdy=2048):
    """
    Sample points on the boundary of [0,1]^2 uniformly.
    Returns:
      X_b: coordinates,
      u_b: zero Dirichlet for u,
      a_b: true a at boundary points.
    """
    pts = []
    for _ in range(num_bdy):
        side = np.random.choice([0,1,2,3])
        if side == 0:
            x_, y_ = 0.0, np.random.rand()
        elif side == 1:
            x_, y_ = 1.0, np.random.rand()
        elif side == 2:
            x_, y_ = np.random.rand(), 0.0
        else:
            x_, y_ = np.random.rand(), 1.0
        pts.append([x_, y_])
    Xb = np.array(pts)
    ab = true_a(Xb[:,0], Xb[:,1]).reshape(-1,1)
    ub = np.zeros_like(ab)
    return torch.tensor(Xb, dtype=torch.float32), torch.tensor(ub, dtype=torch.float32), torch.tensor(ab, dtype=torch.float32)

# 6. Generate high-resolution test data

def generate_test_data(nx=100, ny=100):
    """
    Uniform grid for evaluation of u and a on [0,1]^2.
    Returns X_test, u_test, a_test tensors.
    """
    x_vals = np.linspace(0,1,nx)
    y_vals = np.linspace(0,1,ny)
    xx, yy = np.meshgrid(x_vals, y_vals)
    X = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    ut = true_u(xx.ravel(), yy.ravel()).reshape(-1,1)
    at = true_a(xx.ravel(), yy.ravel()).reshape(-1,1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(ut, dtype=torch.float32), torch.tensor(at, dtype=torch.float32)

# 7. Wrapper to match config-based interface

def generate_data(cfg):
    """
    Read parameters from cfg['data'] and return:
      X_obs, u_obs, X_pde, X_b, u_b, a_b, X_test, u_test, a_test
    """
    d = cfg['data']
    x_obs, u_obs     = generate_mixed_labeled_data(d['grid']['nx'], d['grid']['ny'], d['mix']['noise_noisy'], d['mix']['noise_clean'], 1-d['mix']['ratio_clean'])
    x_pde       = generate_collocation_points(d['pde_points'])
    xb, ub, ab = generate_boundary_points(d['boundary_points'])
    xt, ut, at = generate_test_data(d.get('test_nx',100), d.get('test_ny',100))
    return {
        'x_obs': x_obs, 'u_obs': u_obs,
        'x_pde': x_pde,
        'x_b': xb, 'u_b': ub, 'a_b': ab,
        'x_test': xt, 'u_test': ut, 'a_test': at
    }