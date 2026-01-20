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



def success_rate_ma(x, window_size=100, log_plot=True):
    """Calculate moving average of success rate."""
    time_len = x.shape[-2]
    cumsum = np.cumsum(x, axis=-2)
    cumsum_pad = np.concatenate(
        [np.zeros((*x.shape[:-2], 1, x.shape[-1]), dtype=float), cumsum],
        axis=-2,
    )
    

    # indices for window start and end
    start_idx = np.clip(np.arange(time_len) - window_size + 1, 0, time_len)
    end_idx = np.arange(1, time_len + 1)

    # broadcast indices to match x’s shape
    idx_shape = (1,) * (x.ndim - 2) + (time_len, 1)
    start_idx_b = start_idx.reshape(idx_shape)
    end_idx_b = end_idx.reshape(idx_shape)

    start_take = np.take_along_axis(cumsum_pad, start_idx_b, axis=-2)
    end_take = np.take_along_axis(cumsum_pad, end_idx_b, axis=-2)

    window_sum = end_take - start_take
    window_len = np.minimum(np.arange(time_len) + 1, window_size).reshape(idx_shape)

    return window_sum / window_len



def mean_q(data):
    mean = data.mean(axis=-1)
    lo = np.percentile(data, 0.5, axis=1)
    hi = np.percentile(data, 99.5, axis=1)
    return mean, lo, hi


