import torch
import numpy as np

# 'cuda' device for supported NVIDIA GPU, 'mps' for Apple silicon (M1-M3)
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'mps'
    if torch.backends.mps.is_available() else 'cpu')


def from_numpy(
        x: np.ndarray,
        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert a numpy array to a torch tensor.
    
    Parameters:
        x (np.ndarray): Input numpy array.
        dtype (torch.dtype): Data type of the resulting torch tensor. Default is torch.float32.
    
    Returns:
        torch.Tensor: Torch tensor created from the input numpy array.
    """
    return torch.tensor(x, dtype=dtype, device=device)

def to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor to a numpy array.
    
    Parameters:
        x (torch.Tensor): Input torch tensor.
    
    Returns:
        np.ndarray: Numpy array created from the input torch tensor.
    """
    if x.is_cuda:
        x = x.cpu()
    return x.detach().numpy()
