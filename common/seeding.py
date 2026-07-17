import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed the random number generators used across the training/eval pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(forced_device: str | None = None) -> torch.device:
    """Select the torch device to use: forced_device if given, otherwise the best available (CUDA, MPS, then CPU)."""
    if forced_device:
        return torch.device(forced_device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
