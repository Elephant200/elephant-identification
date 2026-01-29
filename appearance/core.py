"""Core utilities for elephant identification pipeline.

Provides PyTorch device configuration used by training and testing modules.
"""
import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)

# Global device variable set by configure_device()
_device: torch.device | None = None


def configure_device(device: Literal['auto', 'CPU', 'CUDA', 'MPS'] = 'auto') -> str:
    """Configure PyTorch for the specified or auto-detected backend.

    Args:
        device: Backend to use. Options:
            - 'auto': Auto-detect best available (CUDA > MPS > CPU)
            - 'CPU': Force CPU backend
            - 'CUDA': Force NVIDIA CUDA backend
            - 'MPS': Force Apple Metal Performance Shaders backend

    Returns:
        str: The configured device name ('CPU', 'CUDA', or 'MPS')

    Raises:
        RuntimeError: If the specified device is not available
    """
    global _device

    if device == 'auto':
        if torch.cuda.is_available():
            device = 'CUDA'
        elif torch.backends.mps.is_available():
            device = 'MPS'
        else:
            device = 'CPU'
        logger.info(f"Auto-detected device: {device}")

    try:
        if device == 'CUDA':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but no CUDA devices found")
            _device = torch.device('cuda')
            logger.info(f"Configured CUDA GPU: {torch.cuda.get_device_name(0)}")

        elif device == 'MPS':
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but MPS is not available")
            _device = torch.device('mps')
            logger.info("Configured Apple Silicon GPU (MPS backend)")

        elif device == 'CPU':
            _device = torch.device('cpu')
            logger.info("Configuring CPU backend")

        else:
            raise ValueError(f"Unknown device: {device}")

        return device

    except Exception as e:
        logger.error(f"Failed to configure {device}: {e}")
        raise RuntimeError(f"PyTorch {device} configuration failed: {e}")


def get_optimal_batch_size(device: str | None = None) -> int:
    """Determine optimal batch size based on available hardware.

    Args:
        device: Optional device hint ('CPU', 'CUDA', 'MPS'). 
                If None, auto-detects from available hardware.

    Returns:
        int: Optimal batch size for the current hardware configuration
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'CUDA'
        elif torch.backends.mps.is_available():
            device = 'MPS'
        else:
            device = 'CPU'

    if device in ('CUDA', 'MPS', 'GPU'):
        return 64
    else:
        return 16


def get_device() -> torch.device:
    """Get the configured PyTorch device.
    
    Returns:
        torch.device: The configured device, or CPU if not configured.
    """
    global _device
    if _device is None:
        _device = torch.device('cpu')
    return _device
