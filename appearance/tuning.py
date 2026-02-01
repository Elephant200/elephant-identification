"""Hyperparameter tuning script for elephant identification model.

Tests different layer names, pool sizes, and PCA components to find optimal configuration.
Includes preprocessing pipeline so it can be run standalone.
"""
import argparse
import json
import logging
import os

import pandas as pd

from .core import configure_tensorflow
from .model import ElephantIdentifier
from .preprocess import preprocess

from utils import print_with_padding

# ResNet50 layer spatial dimensions (from resnet50_keras_layers.txt)
# Maps layer name prefix to spatial size (H=W for square outputs)
LAYER_SPATIAL_SIZES = {
    'conv1': 112,
    'pool1': 56,
    'conv2': 56,
    'conv3': 28,
    'conv4': 14,
    'conv5': 7,
}


def get_layer_spatial_size(layer_name: str) -> int:
    """Get the spatial dimension for a ResNet50 layer."""
    for prefix, size in LAYER_SPATIAL_SIZES.items():
        if layer_name.startswith(prefix):
            return size
    raise ValueError(f"Unknown layer: {layer_name}")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MODEL_CACHE_DIR = 'cache/appearance/models'


def get_model_path(
    layer_name: str,
    pool_size: int,
    ratio: float,
    min_images: int,
    use_rembg: bool,
    n_components: int
) -> str:
    """Generate a descriptive model filename based on configuration.
    
    Args:
        layer_name: ResNet50 layer name
        pool_size: Max pooling size
        ratio: Train/test split ratio
        min_images: Minimum images per elephant
        use_rembg: Whether background removal was used
        n_components: Number of PCA components
        
    Returns:
        Path to model file in cache/appearance/models/
    """
    rembg_str = "rembg" if use_rembg else "norembg"
    filename = f"split{ratio}_{rembg_str}_min{min_images}_{layer_name}_pool{pool_size}_pca{n_components}.pkl"
    return os.path.join(MODEL_CACHE_DIR, filename)


def run_tuning(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_mapping: dict,
    layer_names: list[str],
    pool_sizes: list[int],
    n_components_list: list[int],
    ratio: float = 0.67,
    min_images: int = 9,
    use_rembg: bool = True,
    top_k_values: list[int] | None = None,
    cache_dir: str = 'cache/appearance/features',
    device: str = 'auto',
    force: bool = False
) -> dict:
    """Run hyperparameter tuning across different configurations.

    Args:
        train_df: Training DataFrame with columns ['filepath', 'name']
        test_df: Test DataFrame with columns ['filepath', 'name']
        class_mapping: Dict mapping elephant name/ID to class index
        layer_names: List of ResNet50 layer names to test
        pool_sizes: List of pool sizes to test
        n_components_list: List of PCA component counts to test
        ratio: Train/test split ratio (for model naming)
        min_images: Minimum images per elephant (for model naming)
        use_rembg: Whether background removal was used (for model naming)
        top_k_values: List of k values for top-k accuracy
        cache_dir: Directory to cache features
        device: TensorFlow device to use
        force: If True, ignore cached features and retrain models

    Returns:
        Dict mapping config name to accuracy results
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    configure_tensorflow(device=device)
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    accuracies = {}

    for layer_name in layer_names:
        spatial_size = get_layer_spatial_size(layer_name)

        for pool_size in pool_sizes:
            # Check if pool size is valid for this layer's spatial dimensions
            if pool_size > spatial_size:
                logger.warning(
                    f"Skipping {layer_name} with pool_size={pool_size}: "
                    f"pool size exceeds spatial dimension ({spatial_size}x{spatial_size})"
                )
                continue

            for n_components in n_components_list:
                config_name = f"{layer_name}_pool_{pool_size}_pca_{n_components}"
                print_with_padding(f"Tuning: {config_name}")

                model_path = get_model_path(
                    layer_name=layer_name,
                    pool_size=pool_size,
                    ratio=ratio,
                    min_images=min_images,
                    use_rembg=use_rembg,
                    n_components=n_components
                )

                # Try to load existing model
                if not force and os.path.exists(model_path):
                    model = ElephantIdentifier.load(model_path)
                else:
                    model = ElephantIdentifier(
                        layer_name=layer_name,
                        pool_size=pool_size,
                        n_components=n_components
                    )

                    model.fit(
                        train_df,
                        class_mapping,
                        cache_dir=f"{cache_dir}/train",
                        force=force
                    )

                    # Save the trained model
                    model.save(model_path)
                    logger.info(f"Saved model to {model_path}")

                results = model.evaluate(
                    test_df,
                    top_k_values=top_k_values,
                    cache_dir=f"{cache_dir}/test",
                    force=force
                )

                accuracies[config_name] = results

                for k, metrics in results.items():
                    logger.info(f"  {k}: {metrics['accuracy']:.3f}")

    return accuracies


def print_results_table(accuracies: dict) -> None:
    """Print tuning results as a formatted table."""
    print_with_padding("Tuning Results")
    
    # header = f"| {'Test Name':<35} | {'Top 1':<20} | {'Top 3':<20} | {'Top 5':<20} | {'Top 10':<20} |"
    # print(header)
    # print(f"| {'-'*35} | {'-'*20} | {'-'*20} | {'-'*20} | {'-'*20} |")

    # for test_name, test_accuracies in accuracies.items():
    #     row = f"| {test_name:<35} | "
    #     for key in ['top_1', 'top_3', 'top_5', 'top_10']:
    #         if key in test_accuracies:
    #             acc = test_accuracies[key]
    #             row += f"{acc['accuracy']*100:.1f}% ({acc['correct']}/{acc['total']}) | "
    #         else:
    #             row += f"{'N/A':<20} | "
    #     print(row)

    # print_with_padding("")

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
    
    # Preprocessing arguments
    preprocess_group = parser.add_argument_group('preprocessing')
    preprocess_group.add_argument(
        '--no-preprocess',
        action='store_true',
        help='Skip preprocessing step (uses existing train/test CSVs and class mapping in --metadata-dir)'
    )
    preprocess_group.add_argument(
        '--input-dir',
        type=str,
        default='dataset/ELPephants',
        help='Directory containing raw elephant images.'
    )
    preprocess_group.add_argument(
        '--output-dir',
        type=str,
        default='dataset/appearance_faces',
        help='Directory to save preprocessed face images'
    )
    preprocess_group.add_argument(
        '--metadata-dir',
        type=str,
        default='dataset/appearance_metadata',
        help='Directory to save train/test CSVs and class mapping'
    )
    preprocess_group.add_argument(
        '--min-images',
        type=int,
        default=9,
        help='Minimum images per elephant to include in dataset'
    )
    preprocess_group.add_argument(
        '--ratio',
        type=float,
        default=0.67,
        help='Train/test split ratio'
    )
    preprocess_group.add_argument(
        '--no-rembg',
        action='store_true',
        help='Skip background removal step (never recomputes SAM masks even with --force)'
    )
    
    # Model/tuning arguments
    tuning_group = parser.add_argument_group('tuning')
    tuning_group.add_argument(
        '--optimal',
        action='store_true',
        help='Use the optimal layer and pool size'
    )
    tuning_group.add_argument(
        '--device',
        type=str,
        choices=['auto', 'CPU', 'CUDA', 'MPS'],
        default='auto',
        help='TensorFlow device to use'
    )
    tuning_group.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Optional path to save results as JSON'
    )
    tuning_group.add_argument(
        '--force',
        action='store_true',
        help='Force recomputation of cached features and models (never recomputes SAM masks)'
    )

    args = parser.parse_args()
    
    use_rembg = not args.no_rembg
    
    # Run preprocessing if input-dir is provided
    if not args.no_preprocess:
        logger.info(f"Running preprocessing on {args.input_dir}...")
        train_df, test_df = preprocess(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            metadata_dir=args.metadata_dir,
            min_images=args.min_images,
            ratio=args.ratio,
            force=False,  # Never force reprocess to protect SAM cache
            skip_rembg=args.no_rembg
        )
        
        # Load class mapping from the generated file
        class_mapping_path = os.path.join(args.metadata_dir, 'class_mapping.json')
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
        logger.info(f"Loaded {len(class_mapping)} class mappings from preprocessing")
    else:
        # Load from CSVs
        train_csv = os.path.join(args.metadata_dir, 'train.csv')
        test_csv = os.path.join(args.metadata_dir, 'test.csv')
        class_mapping_path = os.path.join(args.metadata_dir, 'class_mapping.json')
        
        logger.info("Loading training data...")
        train_df = pd.read_csv(train_csv)
        logger.info(f"Loaded {len(train_df)} training samples")

        logger.info("Loading test data...")
        test_df = pd.read_csv(test_csv)
        logger.info(f"Loaded {len(test_df)} test samples")

        logger.info("Loading class mapping...")
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
        logger.info(f"Loaded {len(class_mapping)} class mappings")

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
        all_pool_sizes = [2, 4, 6, 7, 8, 10]
        all_n_components = [10000]

    accuracies = run_tuning(
        train_df=train_df,
        test_df=test_df,
        class_mapping=class_mapping,
        layer_names=all_layer_names,
        pool_sizes=all_pool_sizes,
        n_components_list=all_n_components,
        ratio=args.ratio,
        min_images=args.min_images,
        use_rembg=use_rembg,
        device=args.device,
        force=args.force
    )

    print_results_table(accuracies)

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(accuracies, f, indent=2)
        logger.info(f"Results saved to {args.output_json}")
