import torch
import numpy as np


# 'cuda' device for supported nvidia GPU, 'mps' for Apple silicon (M1-M3)
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'mps'\
        if torch.backends.mps.is_available() else 'cpu')


def from_numpy(
        x: np.ndarray,
        dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return torch.from_numpy(x).to(device, dtype=dtype)
    # raise NotImplementedError('Convert numpy array to torch tensor here and send to device')


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
    # raise NotImplementedError('Convert torch tensor to numpy array here')
    # HINT: if using GPU, move the tensor to CPU before converting to numpy
