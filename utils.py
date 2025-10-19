import torch

def get_default_device()  -> torch.device:
    # prefer CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.set_default_device(str(device))

    return device

