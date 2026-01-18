"""Core utilities for elephant identification pipeline.

Provides shared functionality for PyTorch configuration, image loading,
and feature extraction using MegaDescriptor.
"""
import logging
import os
import pickle
from typing import List, Literal

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# MegaDescriptor-L-384 expects 384x384 images with [0.5, 0.5, 0.5] normalization
INPUT_SIZE = 384
TRANSFORM = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

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
        return 64  # Slightly smaller than before due to larger model
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


def load_image(image_path: str) -> torch.Tensor:
    """Load and preprocess an image for MegaDescriptor.

    Args:
        image_path: Path to the image file to load

    Returns:
        torch.Tensor: Preprocessed image tensor ready for MegaDescriptor inference,
                      shape (1, 3, 384, 384) with values normalized to [-1, 1]

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be loaded or processed
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = TRANSFORM(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        raise ValueError(f"Failed to load and preprocess image {image_path}: {e}")


def create_feature_extractor() -> torch.nn.Module:
    """Create a feature extractor using MegaDescriptor-L-384.

    Returns:
        torch.nn.Module: MegaDescriptor model for feature extraction
    """
    logger.info("Loading MegaDescriptor-L-384 model...")
    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
    model.eval()
    
    device = get_device()
    model = model.to(device)
    
    logger.debug(f"MegaDescriptor loaded on {device}")
    return model


def extract_features_batch(
    data_df: pd.DataFrame,
    feature_extractor: torch.nn.Module,
    batch_size: int | None = None,
    cache_path: str | None = None,
    force: bool = False
) -> List[np.ndarray]:
    """Extract features from images in batches.

    Args:
        data_df: DataFrame with 'filepath' column containing image paths
        feature_extractor: PyTorch model for feature extraction
        batch_size: Batch size for processing. If None, uses optimal size.
        cache_path: Optional path to cache extracted features
        force: If True, ignore cached features and recompute

    Returns:
        List[np.ndarray]: List of feature arrays for each image
    """
    if data_df.empty:
        raise ValueError("Input DataFrame is empty")

    if cache_path and os.path.exists(cache_path) and not force:
        logger.info(f"Loading cached features from {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache, recomputing: {e}")

    if batch_size is None:
        batch_size = get_optimal_batch_size()

    features: List[np.ndarray] = []
    device = get_device()

    for i in tqdm(range(0, len(data_df), batch_size), desc=f"Extracting features (batch={batch_size})"):
        batch_df = data_df.iloc[i:i + batch_size]
        batch_images = []

        for _, row in batch_df.iterrows():
            try:
                image = load_image(row['filepath'])
                batch_images.append(image)
            except Exception as e:
                logger.warning(f"Failed to load image {row['filepath']}: {e}")
                continue

        if batch_images:
            batch_tensor = torch.cat(batch_images, dim=0).to(device)

            with torch.no_grad():
                batch_features = feature_extractor(batch_tensor)

            # MegaDescriptor outputs (batch, num_features) - already 1D per image
            for feature_vec in batch_features.cpu().numpy():
                features.append(feature_vec)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        logger.debug(f"Saving {len(features)} features to {cache_path}")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
        except Exception as e:
            logger.error(f"Failed to save features cache: {e}")

    return features


def extract_single_image_features(
    image_path: str,
    feature_extractor: torch.nn.Module
) -> np.ndarray:
    """Extract features from a single image.

    Args:
        image_path: Path to the image file
        feature_extractor: PyTorch model for feature extraction

    Returns:
        np.ndarray: Feature array (1D vector)
    """
    device = get_device()
    
    image = load_image(image_path).to(device)
    
    with torch.no_grad():
        feature_vec = feature_extractor(image)

    return feature_vec.cpu().numpy().squeeze()
