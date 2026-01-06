"""Testing and inference script for curvrank identification model.

Provides CLI interface for evaluating on test sets or predicting single images.
"""
import argparse
import json
import logging
import os

from .model import CurvrankIdentifier
from .train import get_images_and_names_from_csv, get_images_and_names_from_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def evaluate(
    model: CurvrankIdentifier,
    test_paths: list[str],
    test_names: list[str],
    top_k_values: list[int] | None = None
) -> dict:
    """Evaluate a trained model on a test dataset.

    Args:
        model: Trained CurvrankIdentifier model
        test_paths: Paths to preprocessed test ear images
        test_names: True names/IDs for each test image
        top_k_values: List of k values for top-k accuracy

    Returns:
        Dict with accuracy metrics for each top-k value
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    accuracies = model.evaluate(test_paths, test_names, top_k_values=top_k_values)

    for k, metrics in accuracies.items():
        logger.info(f"{k}: {metrics['accuracy']:.3f} ({metrics['correct']}/{metrics['total']})")

    return accuracies


def predict_image(
    model: CurvrankIdentifier,
    image_path: str,
    top_k: int | None = None
) -> list[tuple[str, float]]:
    """Predict elephant identity for a single preprocessed ear image.

    Args:
        model: Trained CurvrankIdentifier model
        image_path: Path to the preprocessed ear image
        top_k: If specified, only return top k predictions

    Returns:
        List of (name, score) tuples sorted by score
    """
    predictions = model.predict(image_path)

    if top_k:
        predictions = predictions[:top_k]

    if predictions:
        logger.info(f"Top prediction: {predictions[0][0]} (score: {predictions[0][1]:.4f})")
        for i, (name, score) in enumerate(predictions[:5]):
            logger.debug(f"  #{i+1}: {name} ({score:.4f})")
    else:
        logger.warning("No predictions returned")

    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test or run inference with curvrank identification model'
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
        default='dataset/curvrank_metadata/test.csv',
        help='Path to test CSV file (default: dataset/curvrank_metadata/test.csv)'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default=None,
        help='Directory containing preprocessed test ear images (alternative to --test-csv)'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to single preprocessed ear image for prediction'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        nargs='+',
        default=[1, 3, 5, 10],
        help='List of k values for top-k accuracy (default: 1 3 5 10)'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Optional path to save results as JSON'
    )

    args = parser.parse_args()

    logger.info(f"Loading model from {args.model}...")
    model = CurvrankIdentifier.load(args.model)
    logger.info(f"Loaded: {model}")

    results = None

    if args.image:
        predictions = predict_image(model, args.image)
        results = [{"name": name, "score": score} for name, score in predictions]
    elif args.test_dir:
        logger.info(f"Loading test images from directory {args.test_dir}...")
        test_paths, test_names = get_images_and_names_from_dir(args.test_dir)
        logger.info(f"Found {len(test_paths)} test images")

        results = evaluate(
            model=model,
            test_paths=test_paths,
            test_names=test_names,
            top_k_values=args.top_k
        )
    else:
        logger.info(f"Loading test images from CSV {args.test_csv}...")
        test_paths, test_names = get_images_and_names_from_csv(args.test_csv)
        logger.info(f"Found {len(test_paths)} test images")

        results = evaluate(
            model=model,
            test_paths=test_paths,
            test_names=test_names,
            top_k_values=args.top_k
        )

    if args.output_json and results:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_json}")


