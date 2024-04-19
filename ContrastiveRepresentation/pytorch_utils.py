import torch
import numpy as np


# 'cuda' device for supported NVIDIA GPU, 'mps' for Apple silicon (M1-M3)
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'mps' \
    if torch.backends.mps.is_available() else 'cpu'
)


def from_numpy(x: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert numpy array to torch tensor and send to device.

    Args:
    - x (np.ndarray): Numpy array to convert.
    - dtype (torch.dtype): Desired data type of the tensor.

    Returns:
    - (torch.Tensor): Torch tensor version of the input numpy array.
    """
    return torch.tensor(x, dtype=dtype).to(device)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor to numpy array.

    Args:
    - x (torch.Tensor): Tensor to convert.

    Returns:
    - (np.ndarray): Numpy array version of the input tensor.
    """
    return x.cpu().numpy() if x.is_cuda else x.detach().numpy()


if __name__ == "__main__":
    # Create a numpy array
    np_array = np.array([1, 2, 3], dtype=np.float32)

    # Convert to a torch tensor
    tensor = from_numpy(np_array)

    # Print out the tensor and its device
    print(f"Tensor: {tensor}, Device: {tensor.device}")

    # Convert back to numpy
    converted_np_array = to_numpy(tensor)

    # Print out the numpy array
    print(f"Numpy Array: {converted_np_array}")
