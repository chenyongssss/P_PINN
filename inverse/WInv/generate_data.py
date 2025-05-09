import numpy as np
import torch

def generate_grid_data(nx: int = 201, nt: int = 201):
    """
    Generate full grid data for the wave inverse problem with true speed c=2.0
    u(x,t) = sin(pi*x)*cos(c*pi*t) + 0.5*sin(4*pi*x)*cos(4*c*pi*t)
    Returns:
      X_all: ndarray[N,2], u_all: ndarray[N,1]
    """
    true_c = 2.0
    x_vals = np.linspace(0, 1, nx)
    t_vals = np.linspace(0, 1, nt)
    xx, tt = np.meshgrid(x_vals, t_vals, indexing='ij')
    X_all = np.stack([xx.ravel(), tt.ravel()], axis=-1)
    u_true = (
        np.sin(np.pi * xx) * np.cos(true_c * np.pi * tt)
        + 0.5 * np.sin(4*np.pi * xx) * np.cos(4*true_c * np.pi * tt)
    )
    u_all = u_true.ravel()[:, None]
    return X_all, u_all


def split_train_test(X_all: np.ndarray, u_all: np.ndarray, train_size: int):
    """
    Randomly split full grid into train and test sets.
    """
    N = X_all.shape[0]
    idx = np.random.permutation(N)
    train_idx = idx[:train_size]
    test_idx  = idx[train_size:]
    return (
        X_all[train_idx], u_all[train_idx],
        X_all[test_idx],  u_all[test_idx]
    )


#create mixture data(add noise)
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


def generate_collocation_points(N: int = 2000):
    """
    Uniform random collocation points in [0,1]^2.
    Returns torch.Tensor[N,2]
    """
    pts = np.random.rand(N,2)
    return torch.tensor(pts, dtype=torch.float32)


def generate_initial_points(N: int = 100):
    """
    Initial condition points at t=0: u=sin(pi*x)+0.5*sin(4*pi*x)
    Returns (X_i, u_i) as torch.Tensors
    """
    x = np.linspace(0,1,N)[:,None]
    t = np.zeros_like(x)
    X_i = np.hstack([x, t])
    u_i = np.sin(np.pi*x) + 0.5*np.sin(4*np.pi*x)
    return torch.tensor(X_i, dtype=torch.float32), torch.tensor(u_i, dtype=torch.float32)


def generate_boundary_points(N: int = 200):
    """
    Boundary at x=0 or x=1, u=0, random t in [0,1].
    """
    pts = []
    for _ in range(N):
        side = np.random.choice([0.0,1.0])
        t = np.random.rand()
        pts.append([side, t])
    X_b = np.array(pts)
    u_b = np.zeros((N,1))
    return torch.tensor(X_b, dtype=torch.float32), torch.tensor(u_b, dtype=torch.float32)


def generate_data(cfg: dict) -> dict:
    """
    Generate all datasets according to cfg['data']:
      nx, nt, train_size,
      noisy_ratio, noise_noisy, noise_clean,
      pde_points, boundary_points, initial_points
    Returns dict of torch.Tensors:
      X_data, u_data, X_pde, X_b, u_b, X_i, u_i, X_test, u_test
    """
    d = cfg['data']
    X_all, u_all = generate_grid_data(d['nx'], d['nt'])
    X_tr, u_tr, X_te, u_te = split_train_test(X_all, u_all, d['train_size'])
    X_data_np, u_data_np = create_mixed_labeled_data(
        X_tr, u_tr,
         d['noise_noisy'], d['noise_clean'],1-d['ratio_clean']
    )
    X_pde = generate_collocation_points(d['pde_points'])
    X_b, u_b = generate_boundary_points(d['boundary_points'])
    X_i, u_i = generate_initial_points(d['initial_points'])
    # convert to torch
    to_t = lambda arr: torch.tensor(arr, dtype=torch.float32)
    return {
        'X_data': X_data_np, 'u_data': u_data_np,
        'X_pde' : X_pde,         
        'X_b'   : X_b,           'u_b': u_b,
        'X_i'   : X_i,           'u_i': u_i,
        'X_test': to_t(X_te),    'u_test': to_t(u_te)
    }