import torch
import numpy as np

def l2_relative_error(pred, true):
    """
    Compute the relative L2 error: ||pred - true||_2 / ||true||_2.
    """
    num = torch.norm(pred - true, p=2)
    den = torch.norm(true, p=2) + 1e-8
    return num / den


def l1_relative_error(pred, true):
    """
    Compute the relative L1 error: sum(|pred - true|) / sum(|true|).
    """
    num = torch.sum(torch.abs(pred - true))
    den = torch.sum(torch.abs(true)) + 1e-8
    return num / den


def mse(pred, true):
    """
    Mean squared error.
    """
    return torch.mean((pred - true) ** 2)


def max_error(pred, true):
    """
    Maximum absolute error.
    """
    return torch.max(torch.abs(pred - true))


def residual_norm(residual):
    """
    Compute the L2 norm of a PDE residual tensor, flattening if needed.
    """
    return torch.norm(residual.view(-1), p=2)

def compute_frmse_bands_2d(u_pred, u_true, freq_bands=[(0,4), (5,12), (13,1e9)]):
    """
    Compute the FRMSE in different frequency bands for a 2D field.

    Parameters:
    -----------
    u_pred : 2D numpy array
        Predicted solution on a grid, shape (Nx, Ny).
    u_true : 2D numpy array
        True solution on the same grid, shape (Nx, Ny).
    freq_bands : list of tuples
        Each tuple is (k_min, k_max) defining a frequency range (radial index).
        Default is [(0,4), (5,12), (13,1e9)] for low, middle, and high frequencies.

    Returns:
    --------
    frmses : list of floats
        FRMSE for each band in the order of freq_bands.
    """

    # Dimensions
    Nx, Ny = u_pred.shape

    # 1) 2D FFT
    U_pred = np.fft.fft2(u_pred)
    U_true = np.fft.fft2(u_true)

    # 2) Shift zero frequency to the center
    D = U_pred - U_true
    D_shifted = np.fft.fftshift(D)

    # 3) Build frequency index arrays (k_x, k_y)
    #    np.fft.fftfreq gives frequencies in the range [-0.5, 0.5) * Nx (or Ny) if you multiply.
    freq_x = np.fft.fftfreq(Nx)  # in cycles per grid length
    freq_y = np.fft.fftfreq(Ny)
    # Shift them so that zero freq is at the center
    freq_x = np.fft.fftshift(freq_x) * Nx
    freq_y = np.fft.fftshift(freq_y) * Ny

    # Create 2D mesh for radial frequency
    fx_grid, fy_grid = np.meshgrid(freq_x, freq_y, indexing='ij')
    r = np.sqrt(fx_grid**2 + fy_grid**2)

    # 4) For each band, accumulate squared difference in that frequency range
    frmses = []
    magnitude_sq = np.abs(D_shifted)**2  # squared magnitude of difference in frequency domain

    for (kmin, kmax) in freq_bands:
        mask = (r >= kmin) & (r <= kmax)
        count = np.sum(mask)
        if count > 0:
            band_error = np.sum(magnitude_sq[mask])
            # FRMSE = sqrt( average of squared error in that band )
            frmse = np.sqrt(band_error / count)
        else:
            frmse = 0.0
        frmses.append(frmse)

    return frmses

def evaluate_frmse(u_pred, u_true):
    """
    Example wrapper to compute low/mid/high FRMSE for u_pred vs u_true.

    Returns a dictionary with 'low', 'middle', and 'high' FRMSE values.
    """
    bands = [(0, 4), (5, 12), (13, 1e9)]  # adjust as needed
    frmse_values = compute_frmse_bands_2d(u_pred, u_true, freq_bands=bands)
    return {
        "low_frmse": frmse_values[0],
        "mid_frmse": frmse_values[1],
        "high_frmse": frmse_values[2]
    }

