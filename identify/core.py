"""Core utilities for elephant identification pipeline.

Provides shared functionality for TensorFlow configuration, image loading,
and feature extraction used by both training and testing modules.
"""
import logging
import os
import pickle
from typing import List, Literal

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import ResNet50
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def configure_tensorflow(device: Literal['auto', 'CPU', 'CUDA', 'MPS'] = 'auto') -> str:
    """Configure TensorFlow for the specified or auto-detected backend.

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
    gpus = tf.config.list_physical_devices('GPU')
    
    if device == 'auto':
        if gpus:
            gpu_details = gpus[0].name.lower() if gpus else ''
            if 'nvidia' in gpu_details or any('cuda' in str(g).lower() for g in gpus):
                device = 'CUDA'
            else:
                device = 'MPS'
        else:
            device = 'CPU'
        logger.info(f"Auto-detected device: {device}")

    try:
        if device == 'CUDA':
            if not gpus:
                raise RuntimeError("CUDA requested but no GPU devices found")
            
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Configured {len(gpus)} CUDA GPU(s) with memory growth")
            
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            logger.debug("Enabled mixed precision (float16) for CUDA")

        elif device == 'MPS':
            if not gpus:
                raise RuntimeError("MPS requested but no GPU devices found")
            
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logger.warning(f"MPS memory growth configuration failed: {e}")
            
            logger.info("Configured Apple Silicon GPU (MPS backend)")
            
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            logger.debug("Enabled mixed precision (float16) for MPS")

        elif device == 'CPU':
            logger.info("Configuring CPU backend")
            tf.config.threading.set_intra_op_parallelism_threads(0)
            tf.config.threading.set_inter_op_parallelism_threads(0)
            logger.debug("Set CPU threading to automatic parallelism")

        else:
            raise ValueError(f"Unknown device: {device}")

        return device

    except Exception as e:
        logger.error(f"Failed to configure {device}: {e}")
        raise RuntimeError(f"TensorFlow {device} configuration failed: {e}")


def get_optimal_batch_size(device: str | None = None) -> int:
    """Determine optimal batch size based on available hardware.

    Args:
        device: Optional device hint ('CPU', 'CUDA', 'MPS'). 
                If None, auto-detects from available hardware.

    Returns:
        int: Optimal batch size for the current hardware configuration
    """
    if device is None:
        has_gpu = bool(tf.config.list_physical_devices('GPU'))
        device = 'GPU' if has_gpu else 'CPU'

    if device in ('CUDA', 'MPS', 'GPU'):
        return 128
    else:
        return 32


def get_device_string() -> str:
    """Get the appropriate TensorFlow device string for the current hardware."""
    if tf.config.list_physical_devices('GPU'):
        return '/GPU:0'
    return '/CPU:0'


def load_image(image_path: str) -> tf.Tensor:
    """Load and preprocess an image for ResNet50.

    Args:
        image_path: Path to the image file to load

    Returns:
        tf.Tensor: Preprocessed image tensor ready for ResNet50 inference,
                   shape (1, 224, 224, 3) with values normalized for ImageNet

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be loaded or processed
    """
    try:
        image_raw = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_raw, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, 0)
        image = keras.applications.resnet.preprocess_input(image)
        return image
    except tf.errors.NotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise ValueError(f"Failed to load and preprocess image {image_path}: {e}")


def create_feature_extractor(layer_name: str, pool_size: int = 6) -> keras.Model:
    """Create a feature extractor from ResNet50.

    Args:
        layer_name: Name of the ResNet50 layer to extract features from
        pool_size: Size of the max pooling layer. Defaults to 6.
                   Use 1 to disable pooling.

    Returns:
        keras.Model: Feature extraction model

    Raises:
        ValueError: If the specified layer name doesn't exist in ResNet50
    """
    try:
        model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )

        target_layer = model.get_layer(name=layer_name)

        if pool_size > 1:
            pooled = keras.layers.MaxPooling2D(
                pool_size=(pool_size, pool_size),
                name='feature_pooling'
            )(target_layer.output)
            output = pooled
        else:
            output = target_layer.output

        feature_extractor = keras.Model(
            inputs=model.input,
            outputs=output,
            name='feature_extractor'
        )

        logger.debug(f"Created feature extractor from layer: {layer_name}")
        logger.debug(f"Output shape: {feature_extractor.output_shape}")

        return feature_extractor

    except ValueError as e:
        logger.error(f"Layer '{layer_name}' not found.")
        raise ValueError(f"Invalid layer name '{layer_name}': {e}")


def extract_features_batch(
    data_df: pd.DataFrame,
    feature_extractor: keras.Model,
    batch_size: int | None = None,
    cache_path: str | None = None,
    force: bool = False
) -> List[np.ndarray]:
    """Extract features from images in batches.

    Args:
        data_df: DataFrame with 'filepath' column containing image paths
        feature_extractor: Keras model for feature extraction
        batch_size: Batch size for processing. If None, uses optimal size.
        cache_path: Optional path to cache extracted features
        force: If True, ignore cached features and recompute

    Returns:
        List[np.ndarray]: List of flattened feature arrays for each image
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
    device = get_device_string()

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
            batch_tensor = tf.concat(batch_images, axis=0)

            with tf.device(device):
                batch_features = feature_extractor(batch_tensor)

            for feature_map in batch_features.numpy():
                features.append(feature_map.flatten())

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
    feature_extractor: keras.Model
) -> np.ndarray:
    """Extract features from a single image.

    Args:
        image_path: Path to the image file
        feature_extractor: Keras model for feature extraction

    Returns:
        np.ndarray: Flattened feature array
    """
    device = get_device_string()
    
    with tf.device(device):
        image = load_image(image_path)
        feature_map = feature_extractor(image)

    return feature_map.numpy().flatten()

