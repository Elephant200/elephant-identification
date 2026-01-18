"""Hyperparameter tuning script for elephant identification model.

Tests different layer names, pool sizes, and PCA components to find optimal configuration.
"""
import argparse
import json
import logging

import pandas as pd

from .core import configure_tensorflow
from .model import ElephantIdentifier
from utils import print_with_padding

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def run_tuning(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_mapping: dict,
    layer_names: list[str],
    pool_sizes: list[int],
    n_components_list: list[int],
    top_k_values: list[int] | None = None,
    cache_dir: str = 'cache/appearance/features',
    device: str = 'auto'
) -> dict:
    """Run hyperparameter tuning across different configurations.

    Args:
        train_df: Training DataFrame with columns ['filepath', 'name']
        test_df: Test DataFrame with columns ['filepath', 'name']
        class_mapping: Dict mapping elephant name/ID to class index
        layer_names: List of ResNet50 layer names to test
        pool_sizes: List of pool sizes to test
        n_components_list: List of PCA component counts to test
        top_k_values: List of k values for top-k accuracy
        cache_dir: Directory to cache features
        device: TensorFlow device to use

    Returns:
        Dict mapping config name to accuracy results
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    configure_tensorflow(device=device)

    accuracies = {}

    for layer_name in layer_names:
        for pool_size in pool_sizes:
            for n_components in n_components_list:
                config_name = f"{layer_name}_pool_{pool_size}_pca_{n_components}"
                print_with_padding(f"Tuning: {config_name}")

                model = ElephantIdentifier(
                    layer_name=layer_name,
                    pool_size=pool_size,
                    n_components=n_components
                )

                model.fit(
                    train_df,
                    class_mapping,
                    cache_dir=cache_dir,
                    force=False
                )

                results = model.evaluate(
                    test_df,
                    top_k_values=top_k_values,
                    cache_dir=cache_dir,
                    force=False
                )

                accuracies[config_name] = results

                for k, metrics in results.items():
                    logger.info(f"  {k}: {metrics['accuracy']:.3f}")

    return accuracies


def print_results_table(accuracies: dict) -> None:
    """Print tuning results as a formatted table."""
    print_with_padding("Tuning Results")
    
    header = f"| {'Test Name':<35} | {'Top 1':<20} | {'Top 3':<20} | {'Top 5':<20} | {'Top 10':<20} |"
    print(header)
    print(f"| {'-'*35} | {'-'*20} | {'-'*20} | {'-'*20} | {'-'*20} |")

    for test_name, test_accuracies in accuracies.items():
        row = f"| {test_name:<35} | "
        for key in ['top_1', 'top_3', 'top_5', 'top_10']:
            if key in test_accuracies:
                acc = test_accuracies[key]
                row += f"{acc['accuracy']*100:.1f}% ({acc['correct']}/{acc['total']}) | "
            else:
                row += f"{'N/A':<20} | "
        print(row)

    print_with_padding("")

    compact_header = f"| {'Test Name':<35} | {'Top 1':<10} | {'Top 3':<10} | {'Top 5':<10} | {'Top 10':<10} |"
    print(compact_header)
    print(f"| {'-'*35} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} |")

    for test_name, test_accuracies in accuracies.items():
        row = f"| {test_name:<35} | "
        for key in ['top_1', 'top_3', 'top_5', 'top_10']:
            if key in test_accuracies:
                acc = test_accuracies[key]
                row += f"{acc['accuracy']*100:.1f}%      | "
            else:
                row += f"{'N/A':<10} | "
        print(row)

    print_with_padding("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tune the ResNet50 model for elephant identification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--optimal',
        action='store_true',
        help='Use the optimal layer and pool size'
    )
    parser.add_argument(
        '--train-csv',
        type=str,
        default='dataset/appearance_metadata/train.csv',
        help='Path to training CSV'
    )
    parser.add_argument(
        '--test-csv',
        type=str,
        default='dataset/appearance_metadata/test.csv',
        help='Path to test CSV'
    )
    parser.add_argument(
        '--class-mapping',
        type=str,
        default='dataset/appearance_metadata/class_mapping.json',
        help='Path to class mapping JSON'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'CPU', 'CUDA', 'MPS'],
        default='auto',
        help='TensorFlow device to use'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Optional path to save results as JSON'
    )

    args = parser.parse_args()

    if args.optimal:
        all_layer_names = ["conv4_block6_out"]
        all_pool_sizes = [6]
        all_n_components = [10000]
    else:
        all_layer_names = [
            "conv3_block4_2_relu",
            "conv4_block6_out",
            "conv5_block1_out",
        ]
        all_pool_sizes = [1, 2, 4, 6]
        all_n_components = [10000]

    logger.info("Loading training data...")
    train_df = pd.read_csv(args.train_csv)
    logger.info(f"Loaded {len(train_df)} training samples")

    logger.info("Loading test data...")
    test_df = pd.read_csv(args.test_csv)
    logger.info(f"Loaded {len(test_df)} test samples")

    logger.info("Loading class mapping...")
    with open(args.class_mapping, 'r') as f:
        class_mapping = json.load(f)
    logger.info(f"Loaded {len(class_mapping)} class mappings")

    accuracies = run_tuning(
        train_df=train_df,
        test_df=test_df,
        class_mapping=class_mapping,
        layer_names=all_layer_names,
        pool_sizes=all_pool_sizes,
        n_components_list=all_n_components,
        device=args.device
    )

    print_results_table(accuracies)

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(accuracies, f, indent=2)
        logger.info(f"Results saved to {args.output_json}")
