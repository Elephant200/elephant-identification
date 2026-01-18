"""Elephant identification package.

Provides tools for training and running elephant identification models
using MegaDescriptor feature extraction, PCA, and SVM classification.
"""
from .model import ElephantIdentifier
from .core import configure_device, load_image, create_feature_extractor

__all__ = [
    'ElephantIdentifier',
    'configure_device',
    'load_image',
    'create_feature_extractor',
]
