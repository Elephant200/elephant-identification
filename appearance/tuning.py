"""Hyperparameter tuning script for elephant identification model.

Tests different model configurations to find optimal hyperparameters.
Supports both ResNet50 and MegaDescriptor feature extractors.
"""
import argparse
import json
import logging
import os

import pandas as pd

from .core import configure_device
from .model import ElephantIdentifier, ResNet50Identifier, MegaDescriptorIdentifier
from .preprocess import preprocess

from utils import print_with_padding

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MODEL_CACHE_DIR = 'cache/appearance/models'


def get_model_path(
    model_type: str,
    ratio: float,
    min_images: int,
    use_rembg: bool,
    n_components: int,
    layer_name: str | None = None,
    pool_size: int | None = None
) -> str:
    """Generate a descriptive model filename based on configuration.
    
    Args:
        model_type: 'resnet50' or 'megadescriptor'
        ratio: Train/test split ratio
        min_images: Minimum images per elephant
        use_rembg: Whether background removal was used
        n_components: Number of PCA components
        layer_name: ResNet50 layer name (only for resnet50)
        pool_size: Max pooling size (only for resnet50)
        
    Returns:
        Path to model file in cache/appearance/models/
    """
    rembg_str = "rembg" if use_rembg else "norembg"
    
    if model_type == 'resnet50':
        filename = f"resnet50_split{ratio}_{rembg_str}_min{min_images}_{layer_name}_pool{pool_size}_pca{n_components}.pkl"
    else:
        filename = f"megadescriptor_split{ratio}_{rembg_str}_min{min_images}_pca{n_components}.pkl"
    
    return os.path.join(MODEL_CACHE_DIR, filename)


def run_resnet50_tuning(
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
    force: bool = False
) -> dict:
    """Run hyperparameter tuning for ResNet50 model.

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
        force: If True, ignore cached features and retrain models

    Returns:
        Dict mapping config name to accuracy results
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    accuracies = {}

    for layer_name in layer_names:
        spatial_size = ResNet50Identifier.LAYER_SPATIAL_SIZES.get(layer_name, 14)

        for pool_size in pool_sizes:
            # Check if pool size is valid for this layer's spatial dimensions
            if pool_size > spatial_size:
                logger.warning(
                    f"Skipping {layer_name} with pool_size={pool_size}: "
                    f"pool size exceeds spatial dimension ({spatial_size}x{spatial_size})"
                )
                continue

            for n_components in n_components_list:
                config_name = f"resnet50_{layer_name}_pool{pool_size}_pca{n_components}"
                print_with_padding(f"Tuning: {config_name}")

                model_path = get_model_path(
                    model_type='resnet50',
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
                    model = ResNet50Identifier(
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


def run_megadescriptor_tuning(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_mapping: dict,
    n_components_list: list[int],
    ratio: float = 0.67,
    min_images: int = 9,
    use_rembg: bool = True,
    top_k_values: list[int] | None = None,
    cache_dir: str = 'cache/appearance/features',
    force: bool = False
) -> dict:
    """Run hyperparameter tuning for MegaDescriptor model.

    Args:
        train_df: Training DataFrame with columns ['filepath', 'name']
        test_df: Test DataFrame with columns ['filepath', 'name']
        class_mapping: Dict mapping elephant name/ID to class index
        n_components_list: List of PCA component counts to test
        ratio: Train/test split ratio (for model naming)
        min_images: Minimum images per elephant (for model naming)
        use_rembg: Whether background removal was used (for model naming)
        top_k_values: List of k values for top-k accuracy
        cache_dir: Directory to cache features
        force: If True, ignore cached features and retrain models

    Returns:
        Dict mapping config name to accuracy results
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    accuracies = {}

    for n_components in n_components_list:
        config_name = f"megadescriptor_pca{n_components}"
        print_with_padding(f"Tuning: {config_name}")

        model_path = get_model_path(
            model_type='megadescriptor',
            ratio=ratio,
            min_images=min_images,
            use_rembg=use_rembg,
            n_components=n_components
        )

        # Try to load existing model
        if not force and os.path.exists(model_path):
            model = ElephantIdentifier.load(model_path)
        else:
            model = MegaDescriptorIdentifier(n_components=n_components)

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

    compact_header = f"| {'Test Name':<40} | {'Top 1':<10} | {'Top 3':<10} | {'Top 5':<10} | {'Top 10':<10} |"
    print(compact_header)
    print(f"| {'-'*40} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} |")

    for test_name, test_accuracies in accuracies.items():
        row = f"| {test_name:<40} | "
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
        description='Tune elephant identification models (ResNet50 or MegaDescriptor)',
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
        '--model',
        type=str,
        choices=['resnet50', 'megadescriptor'],
        required=True,
        help='Model type to tune (required)'
    )
    tuning_group.add_argument(
        '--optimal',
        action='store_true',
        help='Use optimal hyperparameters (ResNet50: layer3/pool6, MegaDescriptor: pca1024)'
    )
    tuning_group.add_argument(
        '--n-components',
        type=int,
        nargs='+',
        default=None,
        help='List of PCA component counts to test (default depends on model)'
    )
    tuning_group.add_argument(
        '--device',
        type=str,
        choices=['auto', 'CPU', 'CUDA', 'MPS'],
        default='auto',
        help='PyTorch device to use'
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
    configure_device(device=args.device)
    
    # Run preprocessing if needed
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

    # Run tuning based on model type
    if args.model == 'resnet50':
        if args.optimal:
            layer_names = ['layer3']
            pool_sizes = [6]
            n_components_list = args.n_components or [10000]
        else:
            layer_names = ['layer2', 'layer3', 'layer4']
            pool_sizes = [2, 4, 6, 7]
            n_components_list = args.n_components or [10000]

        accuracies = run_resnet50_tuning(
            train_df=train_df,
            test_df=test_df,
            class_mapping=class_mapping,
            layer_names=layer_names,
            pool_sizes=pool_sizes,
            n_components_list=n_components_list,
            ratio=args.ratio,
            min_images=args.min_images,
            use_rembg=use_rembg,
            force=args.force
        )
    else:  # megadescriptor
        if args.optimal:
            n_components_list = args.n_components or [1024]
        else:
            n_components_list = args.n_components or [256, 512, 1024]

        accuracies = run_megadescriptor_tuning(
            train_df=train_df,
            test_df=test_df,
            class_mapping=class_mapping,
            n_components_list=n_components_list,
            ratio=args.ratio,
            min_images=args.min_images,
            use_rembg=use_rembg,
            force=args.force
        )

    print_results_table(accuracies)

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(accuracies, f, indent=2)
        logger.info(f"Results saved to {args.output_json}")
