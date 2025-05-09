import numpy as np
import torch

from typing import Dict

# reuse core functions from HINV.py

def true_u(x, y, t):
    """
    Analytical solution u(x,y,t) = exp(-t)*sin(pi*x)*sin(pi*y).
    """
    return np.exp(-t) * np.sin(np.pi*x) * np.sin(np.pi*y)


def true_a(x, y):
    """
    Coefficient field a(x,y) = 2 + sin(pi*x)*sin(pi*y).
    """
    return 2.0 + np.sin(np.pi*x) * np.sin(np.pi*y)


def f_func(x, y, t):
    """
    Source term f(x,y,t) matching the true u and a.
    """
    sinx = np.sin(np.pi*x);
    siny = np.sin(np.pi*y);
    cosx = np.cos(np.pi*x);
    cosy = np.cos(np.pi*y);
    term1 = (4*np.pi**2 - 1) * sinx * siny
    term2 = (np.pi**2) * (2*sinx**2*siny**2 - cosx**2*siny**2 - sinx**2*cosy**2)
    return (term1 + term2) * np.exp(-t)


def generate_data_points(num_samples: int):
    """
    Sample (x,y,t) in [-1,1]^2 x [0,1], compute true u, add Gaussian noise.
    Returns X_data: Tensor[N,3], u_data: Tensor[N,1].
    """
    X = np.empty((num_samples, 3))
    X[:,0] = 2*np.random.rand(num_samples) - 1
    X[:,1] = 2*np.random.rand(num_samples) - 1
    X[:,2] = np.random.rand(num_samples)
    u = true_u(X[:,0], X[:,1], X[:,2])
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(u.reshape(-1,1), dtype=torch.float32)


def generate_collocation_points(num_pde: int):
    """
    Uniform random collocation points in [-1,1]^2 x [0,1] for PDE residual.
    """
    X = np.empty((num_pde,3))
    X[:,0] = 2*np.random.rand(num_pde) - 1
    X[:,1] = 2*np.random.rand(num_pde) - 1
    X[:,2] = np.random.rand(num_pde)
    return torch.tensor(X, dtype=torch.float32)


def generate_a_boundary_points(num_bdy: int):
    """
    Points on spatial boundary for a (x=±1 or y=±1), random t in [0,1]; a=2.
    """
    pts, vals = [], []
    for _ in range(num_bdy):
        side = np.random.choice([0,1,2,3])
        t_ = np.random.rand()
        if side==0: x_,y_ = -1, 2*np.random.rand()-1
        elif side==1: x_,y_ = 1,  2*np.random.rand()-1
        elif side==2: x_,y_ = 2*np.random.rand()-1, -1
        else:          x_,y_ = 2*np.random.rand()-1, 1
        pts.append([x_,y_,t_]); vals.append(2.0)
    return (torch.tensor(pts, dtype=torch.float32), torch.tensor(np.array(vals).reshape(-1,1), dtype=torch.float32))


def generate_u_boundary_points(num_bdy: int):
    """
    Points on spatial boundary for u; true Dirichlet: u=exp(-t)*sin(pi x)*sin(pi y).
    """
    pts, vals = [], []
    for _ in range(num_bdy):
        side = np.random.choice([0,1,2,3])
        t_ = np.random.rand()
        if side==0: x_,y_ = -1, 2*np.random.rand()-1
        elif side==1: x_,y_ = 1,  2*np.random.rand()-1
        elif side==2: x_,y_ = 2*np.random.rand()-1, -1
        else:          x_,y_ = 2*np.random.rand()-1, 1
        pts.append([x_,y_,t_]); vals.append(true_u(x_,y_,t_))
    return (torch.tensor(pts, dtype=torch.float32), torch.tensor(np.array(vals).reshape(-1,1), dtype=torch.float32))


def generate_u_initial_points(num_ini: int):
    """
    Initial condition for u at t=0: u=sin(pi x)*sin(pi y).
    """
    pts, vals = [], []
    for _ in range(num_ini):
        x_,y_ = 2*np.random.rand()-1, 2*np.random.rand()-1
        pts.append([x_,y_,0.0]); vals.append(np.sin(np.pi*x_)*np.sin(np.pi*y_))
    return (torch.tensor(pts, dtype=torch.float32), torch.tensor(np.array(vals).reshape(-1,1), dtype=torch.float32))

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
    u_obs = u_d.detach().clone()
    
    # 
    u_obs[clean_idx] += noise_clean * torch.randn_like(u_obs[clean_idx])
    u_obs[noisy_idx] += noise_noisy * torch.randn_like(u_obs[noisy_idx])
    return x_d, u_obs


def generate_data(cfg: Dict) -> Dict:
    """
    Wrapper to produce all required datasets from config.
    cfg['data'] must contain:
      num_data, noise_level, pde_points, boundary_points, initial_points
    """
    d = cfg['data']
    Xd, ud      = generate_data_points(d['samples'])
    x_obs, u_obs = create_mixed_labeled_data(Xd, ud, d['noise_noisy'], d['noise_clean'], 1-d['ratio_clean'])
    Xp          = generate_collocation_points(d['pde_points'])
    Xba, ab     = generate_a_boundary_points(d['boundary_points'])
    Xbu, ub     = generate_u_boundary_points(d['boundary_points'])
    Xiu, ui     = generate_u_initial_points(d['initial_points'])
    return {
        'x_obs': x_obs, 'u_obs': u_obs,
        'X_pde' : Xp,
        'X_b_a' : Xba, 'a_b' : ab,
        'X_b_u' : Xbu, 'u_b' : ub,
        'X_i_u' : Xiu, 'u_i' : ui
    }