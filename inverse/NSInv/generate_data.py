import numpy as np
import scipy.io
import torch
import os

# 1. Load and preprocess data from .mat file

def load_cfd_data(mat_path: str):
    """
    Load CFD data from MATLAB file and filter by domain:
      x in [1,8], y in [-2,2], t in [0,7].
    Returns numpy array of shape (M,6): [x,y,t,u,v,p].
    """
    data = scipy.io.loadmat(mat_path)
    U_star = data['U_star']  # (N,2,T)
    P_star = data['p_star']  # (N,T)
    t_star = data['t']       # (T,1)
    X_star = data['X_star']  # (N,2)
    N, T = X_star.shape[0], t_star.shape[0]

    # Tile coordinates and flatten
    XX = np.tile(X_star[:,0:1], (1, T))
    YY = np.tile(X_star[:,1:2], (1, T))
    TT = np.tile(t_star.T, (N,1))
    UU = U_star[:,0,:]
    VV = U_star[:,1,:]
    PP = P_star

    data_all = np.stack([XX.flatten(), YY.flatten(), TT.flatten(),
                         UU.flatten(), VV.flatten(), PP.flatten()], axis=1)
    # Filter by specified domain
    mask = (data_all[:,0] >= 1) & (data_all[:,0] <= 8) & \
           (data_all[:,1] >= -2) & (data_all[:,1] <= 2) & \
           (data_all[:,2] >= 0) & (data_all[:,2] <= 7)
    return data_all[mask]

# 2. Split into training and testing sets

def split_train_test(data: np.ndarray, N_train: int):
    """
    Shuffle and split data into N_train samples for training,
    rest for testing.
    Training: inputs [x,y,t], targets [u,v].
    Testing: inputs [x,y,t], targets [u,v,p].
    """
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    train_idx = idx[:N_train]
    test_idx  = idx[N_train:]

    train = data[train_idx]
    test  = data[test_idx]
    X_train = train[:, :3]
    uv_train = train[:, 3:5]
    X_test  = test[:, :3]
    uv_test = test[:, 3:5]
    p_test  = test[:, 5:6]
    return (X_train, uv_train), (X_test, uv_test, p_test)

# 3. Generate PDE collocation points in domain

def generate_collocation_points(num_pde: int):
    """
    Sample random points (x,y,t) in the target domain [1,8]x[-2,2]x[0,7].
    """
    x = np.random.uniform(1, 8, size=(num_pde,1))
    y = np.random.uniform(-2,2, size=(num_pde,1))
    t = np.random.uniform(0,7, size=(num_pde,1))
    X_pde = np.hstack([x,y,t])
    return torch.tensor(X_pde, dtype=torch.float32)

#create mixture data(add noise)
def create_mixed_labeled_data(x_d, u_d, noise_noisy=1.0, noise_clean=0.0, noise_ratio=2/5):
    """
    Generate a 50x50 grid on [0,1]^2 with 2500 points.
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

# 5. Wrapper to match config-based interface

def generate_data(cfg: dict):
    """
    Reads configuration keys:
      mat_path, N_train, pde_points
    Returns dict with:
      X_data, uv_data, X_pde, X_test, uv_test, p_test
    """
    d = cfg['data']
    
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 
    mat_path = os.path.join(script_dir, d['mat_path'])  # 
    # print(d['mat_path'])
    raw = load_cfd_data(mat_path)
    
    (X_data_np, uv_data_np), (X_test_np, uv_test_np, p_test_np) = \
        split_train_test(raw, d['N_train'])
    x_obs, u_obs = create_mixed_labeled_data(X_data_np, uv_data_np, noise_noisy=d['noise_noisy'], noise_clean=d['noise_clean'], noise_ratio=1-d['ratio_clean'])
    return {
        'x_obs': x_obs,
        'u_obs': u_obs,
        'x_pde': generate_collocation_points(d['pde_points']),
        'x_test': torch.tensor(X_test_np, dtype=torch.float32),
        'uv_test': torch.tensor(uv_test_np, dtype=torch.float32),
        'p_test': torch.tensor(p_test_np, dtype=torch.float32)
    }