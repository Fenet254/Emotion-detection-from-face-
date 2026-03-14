import torch
print(torch.__version__)
print("CUDA:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))