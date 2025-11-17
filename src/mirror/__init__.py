import torch

"""Top-level module for mirror."""

__author__ = """Fred Shone"""
__email__ = "fredjshone@gmail.com"
__version__ = "0.0.0"


def cuda_available():
    """Check if CUDA is available."""
    try:
        return torch.cuda.is_available()
    except AssertionError:
        return False


def current_device():
    """Get current device."""
    return torch.cuda.current_device() if cuda_available() else "cpu"


if cuda_available():
    torch.set_float32_matmul_precision("medium")
