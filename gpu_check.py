import torch

def estimate_max_batch_size(device_id=0):
    torch.cuda.set_device(device_id)  # Switch to the chosen GPU
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"GPU {device_id} -> Free Memory: {free_mem / 1e6} MB, Total Memory: {total_mem / 1e6} MB")

# Example: Check memory on GPU 1
estimate_max_batch_size(device_id=1)
