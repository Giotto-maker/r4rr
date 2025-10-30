import torch
import random
import numpy as np


def set_seed(seed=0):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int, optional): The seed value to use for random number generators. Defaults to 0.

    Notes:
        - Sets seeds for Python's `random`, NumPy, and PyTorch (CPU and CUDA).
        - Configures PyTorch's cuDNN backend for deterministic behavior.
        - Disables cuDNN benchmarking to further ensure reproducibility.
    """
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def my_gpu_info():
    print("Torch version: ", torch.__version__)  

    if torch.cuda.is_available():
        print("CUDA version: ", torch.version.cuda)
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        
        for i in range(num_gpus):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
    else:
        print("CUDA is not available on this system.")