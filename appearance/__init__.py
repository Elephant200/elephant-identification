"""Elephant identification package.

Provides tools for training and running elephant identification models
using ResNet50 or MegaDescriptor feature extraction, PCA, and SVM classification.
"""
from .model import ElephantIdentifier, ResNet50Identifier, MegaDescriptorIdentifier
from .core import configure_device

__all__ = [
    'ElephantIdentifier',
    'ResNet50Identifier',
    'MegaDescriptorIdentifier',
    'configure_device',
]
