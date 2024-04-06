import torch
import numpy as np

# 'cuda' device for supported NVIDIA GPU, 'mps' for Apple silicon (M1-M3)
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'mps'
    if torch.backends.mps.is_available() else 'cpu')

def from_numpy(x: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Converts a NumPy array to a PyTorch tensor and moves it to the specified device.
    """
    return torch.tensor(x, dtype=dtype, device=device)

def to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array. If the tensor is on a GPU, it will be moved to CPU first.
    """
    return x.detach().cpu().numpy()
