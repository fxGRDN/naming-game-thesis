import torch
import numpy as np

def get_default_device()  -> torch.device:
    # prefer CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.set_default_device(str(device))

    return device


def mean_q(data):
    mean = data.mean(axis=-1)
    lo = np.percentile(data, 0.5, axis=1)
    hi = np.percentile(data, 99.5, axis=1)
    return mean, lo, hi


