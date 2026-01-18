"""Training script for elephant identification model.

Provides CLI interface for training the ElephantIdentifier model.
"""
import argparse
import json
import logging
import os

import pandas as pd

from .core import configure_device
from .model import ElephantIdentifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def train(
    train_df: pd.DataFrame,
    class_mapping: dict,
    output_path: str,
    n_components: int = 1024,
    batch_size: int | None = None,
    cache_dir: str | None = None,
    force: bool = False,
    device: str = 'auto'
) -> ElephantIdentifier:
    """Train an ElephantIdentifier model and save it.

    Args:
        train_df: DataFrame with columns ['filepath', 'name']
        class_mapping: Dict mapping elephant name/ID to class index
        output_path: Path to save the trained model
        n_components: Number of PCA components
        batch_size: Batch size for feature extraction. Auto if None.
        cache_dir: Directory to cache intermediate results
        force: If True, retrain from scratch ignoring cache
        device: PyTorch device ('auto', 'CPU', 'CUDA', 'MPS')

    Returns:
        ElephantIdentifier: The trained model
    """
    configure_device(device=device)

    model = ElephantIdentifier(n_components=n_components)

    model.fit(
        train_df,
        class_mapping,
        batch_size=batch_size,
        cache_dir=cache_dir,
        force=force
    )

    model.save(output_path)
    logger.info(f"Training complete. Model saved to {output_path}")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train elephant identification model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--train-csv',
        type=str,
        default="dataset/appearance_metadata/train.csv",
        help='Path to training CSV file with columns [filepath, name]'
    )
    parser.add_argument(
        '--class-mapping',
        type=str,
        default="dataset/appearance_metadata/class_mapping.json",
        help='Path to class mapping JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='cache/appearance/models/elephant_model.pkl',
        help='Path to save trained model'
    )
    parser.add_argument(
        '--n-components',
        type=int,
        default=1024,
        help='Number of PCA components (MegaDescriptor outputs 1536-dim features)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for feature extraction (auto-detected if not specified)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default="cache/appearance/features/train",
        help='Directory to cache features'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force retrain ignoring cached data'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'CPU', 'CUDA', 'MPS'],
        default='auto',
        help='PyTorch device to use'
    )

    args = parser.parse_args()

    logger.info("Loading training data...")
    train_df = pd.read_csv(args.train_csv)
    logger.info(f"Loaded {len(train_df)} training samples")

    logger.info("Loading class mapping...")
    with open(args.class_mapping, 'r') as f:
        class_mapping = json.load(f)
    logger.info(f"Loaded {len(class_mapping)} class mappings")

    train(
        train_df=train_df,
        class_mapping=class_mapping,
        output_path=args.output,
        n_components=args.n_components,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        force=args.force,
        device=args.device
    )
