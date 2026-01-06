"""Training script for curvrank identification model.

Provides CLI interface for training the CurvrankIdentifier model.
"""
import argparse
import logging
import os

import pandas as pd

from .model import CurvrankIdentifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def train(
    image_paths: list[str],
    names: list[str],
    output_path: str,
    index_dir: str = 'curvrank/.cache/indices'
) -> CurvrankIdentifier:
    """Train a CurvrankIdentifier model and save it.

    Args:
        image_paths: Paths to preprocessed ear images (*_left.jpg or *_right.jpg)
        names: Individual names/IDs corresponding to each image
        output_path: Path to save the trained model
        index_dir: Directory to save LNBNN index files

    Returns:
        CurvrankIdentifier: The trained model
    """
    model = CurvrankIdentifier()

    model.fit(image_paths, names, index_dir=index_dir)

    model.save(output_path)
    logger.info(f"Training complete. Model saved to {output_path}")

    return model


def get_images_and_names_from_dir(preprocessed_dir: str) -> tuple[list[str], list[str]]:
    """Get image paths and names from a preprocessed directory.

    Assumes filenames are in format: {name}_{view}.jpg (e.g., 123_left.jpg)

    Args:
        preprocessed_dir: Directory containing preprocessed ear images

    Returns:
        Tuple of (image_paths, names)
    """
    image_paths = []
    names = []

    for filename in os.listdir(preprocessed_dir):
        if not filename.endswith('.jpg'):
            continue

        filepath = os.path.join(preprocessed_dir, filename)
        base = filename.rsplit('.', 1)[0]
        name = base.split('_')[0]

        image_paths.append(filepath)
        names.append(name)

    return image_paths, names


def get_images_and_names_from_csv(csv_path: str) -> tuple[list[str], list[str]]:
    """Get image paths and names from a CSV file.

    Expects columns: elephant_id, filepath, view

    Args:
        csv_path: Path to CSV file

    Returns:
        Tuple of (image_paths, names)
    """
    df = pd.read_csv(csv_path)
    image_paths = df['filepath'].tolist()
    names = df['elephant_id'].astype(str).tolist()
    return image_paths, names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train curvrank elephant identification model'
    )
    parser.add_argument(
        '--train-csv',
        type=str,
        default='dataset/curvrank_metadata/train.csv',
        help='Path to training CSV file (default: dataset/curvrank_metadata/train.csv)'
    )
    parser.add_argument(
        '--preprocessed-dir',
        type=str,
        default=None,
        help='Directory containing preprocessed ear images (alternative to --train-csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/curvrank_model.pkl',
        help='Path to save trained model (default: models/curvrank_model.pkl)'
    )
    parser.add_argument(
        '--index-dir',
        type=str,
        default='curvrank/.cache/indices',
        help='Directory to save LNBNN index files (default: curvrank/.cache/indices)'
    )

    args = parser.parse_args()

    if args.preprocessed_dir:
        logger.info(f"Loading images from directory {args.preprocessed_dir}...")
        image_paths, names = get_images_and_names_from_dir(args.preprocessed_dir)
    else:
        logger.info(f"Loading images from CSV {args.train_csv}...")
        image_paths, names = get_images_and_names_from_csv(args.train_csv)

    logger.info(f"Found {len(image_paths)} preprocessed ear images")

    unique_names = set(names)
    logger.info(f"Unique individuals: {len(unique_names)}")

    train(
        image_paths=image_paths,
        names=names,
        output_path=args.output,
        index_dir=args.index_dir
    )


