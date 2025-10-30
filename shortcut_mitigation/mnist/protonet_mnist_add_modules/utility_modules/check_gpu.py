import os
import torch


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