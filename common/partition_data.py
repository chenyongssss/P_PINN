import torch
import numpy as np
import inspect

def partition_data(x_obs: torch.Tensor,
                   u_obs: torch.Tensor,
                   model: torch.nn.Module,
                   pde_residual_fn: callable = None,
                   eps: float = 0.01,
                   w_data: float = 1.0,
                   w_res: float = 1.0):
    """
    Split observed data into 'retain' vs. 'forget' sets based on composite score:
      score_i = w_data * ||u_pred_i - u_obs_i|| + w_res * ||residual_i||

    Args:
        x_obs (Tensor[N,in_dim]):  input coordinates of observations
        u_obs (Tensor[N,out_dim]): observed solution values
        model (nn.Module):         a PINN instance
        pde_residual_fn (callable):fn(model, x) -> residual Tensor[N, ...]
        eps (float):               threshold to split
        w_data (float):            weight on data‐error term
        w_res (float):             weight on residual‐error term

    Returns:
        retain_data
        forget_data
    """
    model.eval()
    x_obs = x_obs.clone().detach().requires_grad_(True)
    
    u_pred = model(x_obs)
    

    # data‐error norm per sample
    data_err = ((u_pred - u_obs)**2).sum(dim=1)
        
    if pde_residual_fn is not None:
        res = pde_residual_fn(model, x_obs)
        res_err = (res**2).sum(dim=1)
    else:
        res_err = torch.zeros_like(data_err)

    scores = w_data * data_err + w_res * res_err
    scores = scores.detach().cpu().numpy().squeeze()
    
    good_idx = np.where(scores < eps)[0]
    bad_idx = np.where(scores >= eps)[0]
    print(f"Composite loss threshold: {eps:.6e}, Good data count: {len(good_idx)}, Bad data count: {len(bad_idx)}")
    X_good = x_obs[good_idx].detach()
    u_good = u_obs[good_idx]
    X_bad = x_obs[bad_idx].detach()
    u_bad = u_obs[bad_idx]
    return (X_good, u_good), (X_bad, u_bad)


