"""Testing and inference script for elephant identification model.

Provides CLI interface for evaluating on test sets or predicting single images.
"""
import argparse
import json
import logging
from typing import List

import pandas as pd

from .core import configure_tensorflow
from .model import ElephantIdentifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def evaluate(
    model: ElephantIdentifier,
    test_df: pd.DataFrame,
    batch_size: int | None = None,
    top_k_values: List[int] | None = None,
    cache_dir: str | None = None,
    force: bool = False
) -> dict:
    """Evaluate a trained model on a test dataset.

    Args:
        model: Trained ElephantIdentifier model
        test_df: DataFrame with columns ['filepath', 'name']
        batch_size: Batch size for feature extraction
        top_k_values: List of k values for top-k accuracy
        cache_dir: Directory to cache test features
        force: If True, ignore cached features

    Returns:
        Dict with accuracy metrics for each top-k value
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    accuracies = model.evaluate(
        test_df,
        batch_size=batch_size,
        top_k_values=top_k_values,
        cache_dir=cache_dir,
        force=force
    )

    for k, metrics in accuracies.items():
        logger.info(f"{k}: {metrics['accuracy']:.3f} ({metrics['correct']}/{metrics['total']})")

    return accuracies


def predict_image(
    model: ElephantIdentifier,
    image_path: str,
    top_k: int | None = None
) -> List[tuple]:
    """Predict elephant identity for a single image.

    Args:
        model: Trained ElephantIdentifier model
        image_path: Path to the image file
        top_k: If specified, only return top k predictions

    Returns:
        List of (name, confidence) tuples sorted by confidence descending
    """
    predictions = model.predict(image_path)
    
    if top_k:
        predictions = predictions[:top_k]

    logger.info(f"Top prediction: {predictions[0][0]} (confidence: {predictions[0][1]:.3f})")
    for i, (name, conf) in enumerate(predictions[:5]):
        logger.debug(f"  #{i+1}: {name} ({conf:.3f})")

    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test or run inference with elephant identification model'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.pkl)'
    )
    parser.add_argument(
        '--test-csv',
        type=str,
        default=None,
        help='Path to test CSV file for evaluation'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to single image for prediction'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        nargs='+',
        default=[1, 3, 5, 10],
        help='List of k values for top-k accuracy (default: 1 3 5 10)'
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
        default='appearance/.cache',
        help='Directory to cache features (default: appearance/.cache)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recompute ignoring cached features'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'CPU', 'CUDA', 'MPS'],
        default='auto',
        help='TensorFlow device to use (default: auto)'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Optional path to save results as JSON'
    )

    args = parser.parse_args()

    if not args.test_csv and not args.image:
        parser.error("Must specify either --test-csv or --image")

    configure_tensorflow(device=args.device)

    logger.info(f"Loading model from {args.model}...")
    model = ElephantIdentifier.load(args.model)
    logger.info(f"Loaded: {model}")

    results = None

    if args.test_csv:
        logger.info(f"Loading test data from {args.test_csv}...")
        test_df = pd.read_csv(args.test_csv)
        logger.info(f"Loaded {len(test_df)} test samples")

        results = evaluate(
            model=model,
            test_df=test_df,
            batch_size=args.batch_size,
            top_k_values=args.top_k,
            cache_dir=args.cache_dir,
            force=args.force
        )

    if args.image:
        predictions = predict_image(model, args.image)
        results = [{"name": name, "confidence": conf} for name, conf in predictions]

    if args.output_json and results:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_json}")

