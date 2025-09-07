import argparse
import json
import logging
import os
import pickle
import shutil
import time
from typing import Dict, List, Optional, Tuple

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Only add handler if it doesn't already exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def configure_apple_silicon_gpu() -> None:
    """Configure TensorFlow for optimal Apple Silicon GPU usage.
    
    Sets up Metal Performance Shaders (MPS) backend for Apple Silicon GPUs,
    configures memory growth, and optimizes TensorFlow settings for M1/M2 chips.
    
    Raises:
        RuntimeError: If Apple Silicon GPU configuration fails
    """
    try:
        # Check if MPS is available (Apple Silicon GPU)
        if tf.config.list_physical_devices('GPU'):
            logger.info("Apple Silicon GPU detected, configuring MPS backend")
            
            # Enable memory growth to prevent OOM errors
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.debug(f"Configured {len(gpus)} GPU(s) with memory growth")
                except RuntimeError as e:
                    logger.warning(f"GPU memory growth configuration failed: {e}")
            
            # Set up mixed precision for better performance
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
        else:
            logger.info("No Apple Silicon GPU detected, using CPU")
            # Optimize CPU performance
            tf.config.threading.set_intra_op_parallelism_threads(0)
            tf.config.threading.set_inter_op_parallelism_threads(0)
            
    except Exception as e:
        logger.error(f"Failed to configure Apple Silicon GPU: {e}")
        raise RuntimeError(f"Apple Silicon GPU configuration failed: {e}")


def get_optimal_batch_size() -> int:
    """Determine optimal batch size based on available hardware.
    
    Returns:
        int: Optimal batch size for the current hardware configuration
    """
    if tf.config.list_physical_devices('GPU'):
        # Apple Silicon GPU - use larger batch size
        return 128
    else:
        # CPU fallback - smaller batch size
        return 32


def load_image(image_path: str) -> tf.Tensor:
    """Load and preprocess an image for ResNet50 with GPU optimization.
    
    Args:
        image_path (str): Path to the image file to load
        
    Returns:
        tf.Tensor: Preprocessed image tensor ready for ResNet50 inference,
                  shape (1, 224, 224, 3) with values normalized for ImageNet
                  
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be loaded or processed
    """
    try:
        # Use tf.io for better GPU integration
        image_raw = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_raw, channels=3)
        
        # Resize using tf.image for GPU acceleration
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32)
        
        # Add batch dimension and apply ResNet preprocessing
        image = tf.expand_dims(image, 0)
        image = tf.keras.applications.resnet.preprocess_input(image)
        
        return image
        
    except tf.errors.NotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise ValueError(f"Failed to load and preprocess image {image_path}: {e}")


def extract_raw_features(
    data_df: pd.DataFrame, 
    feature_extractor: keras.Model, 
    layer_name: str = 'conv3_block4_2_relu', 
    cache_dir: str = 'train_cache', 
    force_retrain: bool = False,
    batch_size: Optional[int] = None,
    pool_size: int = 2
) -> Tuple[List[np.ndarray], List[str]]:
    """Extract and cache raw features from images with GPU-optimized batch processing.
    
    Args:
        data_df (pd.DataFrame): DataFrame containing image paths and labels with columns 
                               ['filepath', 'name']
        feature_extractor (keras.Model): Pre-trained model for feature extraction
        layer_name (str): Name of the layer to extract features from. 
                         Defaults to 'conv3_block4_2_relu'
        cache_dir (str): Directory to store cached features. Defaults to 'train_cache'
        force_retrain (bool): If True, ignore cached features and recompute. 
                             Defaults to False
        batch_size (Optional[int]): Batch size for processing. If None, uses optimal size
                                   based on hardware detection
        pool_size (int): Size of the max pooling layer used in feature extraction.
                        Defaults to 2. Used for cache file naming
    
    Returns:
        Tuple[List[np.ndarray], List[str]]: Tuple containing:
            - List of flattened feature arrays for each image
            - List of corresponding labels/names
            
    Raises:
        OSError: If cache directory cannot be created or accessed
        ValueError: If input DataFrame is empty or missing required columns
    """
    # Validate inputs
    if data_df.empty:
        raise ValueError("Input DataFrame is empty")
    
    required_columns = ['filepath', 'name']
    missing_columns = [col for col in required_columns if col not in data_df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = f"{cache_dir}/raw_features_{layer_name}_pool{pool_size}.pkl"

    if os.path.exists(cache_file) and not force_retrain:
        logger.info(f"Loading cached raw features from {cache_file}")
        try:
            return pickle.load(open(cache_file, 'rb'))
        except Exception as e:
            logger.warning(f"Failed to load cache, recomputing: {e}")

    if force_retrain and os.path.exists(cache_file):
        logger.info(f"Force retrain: Removing cached raw features from {cache_file}")
        os.remove(cache_file)
    
    # Use optimal batch size if not specified
    if batch_size is None:
        batch_size = get_optimal_batch_size()
    
    features: List[np.ndarray] = []
    labels: List[str] = []
    
    start_time = time.time()
    
    # Process images in batches for better GPU utilization
    for i in tqdm(range(0, len(data_df), batch_size), desc=f"Extracting raw features with batch size {batch_size}: "):
        batch_df = data_df.iloc[i:i + batch_size]
        batch_images = []
        batch_labels = []
        
        # Load batch of images
        for _, row in batch_df.iterrows():
            try:
                image = load_image(row['filepath'])
                batch_images.append(image)
                batch_labels.append(str(row['name']))
            except Exception as e:
                logger.warning(f"Failed to load image {row['filepath']}: {e}")
                continue
        
        if batch_images:
            # Concatenate images into a single batch tensor
            batch_tensor = tf.concat(batch_images, axis=0)
            
            # Extract features for the entire batch
            with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                batch_features = feature_extractor(batch_tensor)
            
            # Convert to numpy and flatten each feature map
            batch_features_np = batch_features.numpy()
            for j, feature_map in enumerate(batch_features_np):
                features.append(feature_map.flatten())
                labels.append(batch_labels[j])
        
    feature_extraction_time = time.time() - start_time
    logger.info(f"Feature extraction completed in {feature_extraction_time:.2f}s "
               f"({len(data_df)/feature_extraction_time:.1f} images/sec)")

    logger.debug(f"Saving {len(features)} features to {cache_file}")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump((features, labels), f)
    except Exception as e:
        logger.error(f"Failed to save features cache: {e}")
        raise OSError(f"Could not save features to {cache_file}: {e}")
        
    return features, labels


def apply_pca(
    features: List[np.ndarray], 
    n_components: int = 500, 
    cache_dir: str = 'train_cache', 
    layer_name: str = 'conv3_block4_2_relu', 
    force_retrain: bool = False,
    pool_size: int = 2
) -> Tuple[np.ndarray, PCA, StandardScaler]:
    """Apply PCA dimensionality reduction to extracted features.
    
    Args:
        features (List[np.ndarray]): List of flattened feature vectors from images
        n_components (int): Number of principal components to retain. 
                           Defaults to 500
        cache_dir (str): Directory to store cached PCA results. 
                        Defaults to 'train_cache'
        layer_name (str): Name of the feature extraction layer for cache naming. 
                         Defaults to 'conv3_block4_2_relu'
        force_retrain (bool): If True, ignore cached PCA and recompute. 
                             Defaults to False
        pool_size (int): Size of the max pooling layer used in feature extraction.
                        Defaults to 2. Used for cache file naming
    
    Returns:
        Tuple[np.ndarray, PCA, StandardScaler]: Tuple containing:
            - Transformed features with reduced dimensionality (n_samples, n_components)
            - Fitted PCA transformer for future use
            - Fitted StandardScaler for consistent preprocessing
            
    Raises:
        ValueError: If features list is empty or n_components is invalid
        OSError: If cache operations fail
    """
    if not features:
        raise ValueError("Features list is empty")
    
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = f"{cache_dir}/pca_{n_components}_{layer_name}_pool{pool_size}.pkl"

    if os.path.exists(cache_file) and not force_retrain:
        logger.info(f"Loading cached PCA features from {cache_file}")
        try:
            return pickle.load(open(cache_file, 'rb'))
        except Exception as e:
            logger.warning(f"Failed to load PCA cache, recomputing: {e}")

    if force_retrain and os.path.exists(cache_file):
        logger.info(f"Force retrain: Removing cached PCA features from {cache_file}")
        os.remove(cache_file)

    logger.info(f"Applying PCA dimensionality reduction to {len(features)} features...")

    start_time = time.time()
    
    try:
        # Convert to numpy array for sklearn
        X = np.array(features)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        actual_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
        if actual_components != n_components:
            logger.warning(f"Requested {n_components} components, using {actual_components} based on data shape")
        
        pca = PCA(n_components=actual_components)
        X_pca = pca.fit_transform(X_scaled)
        
        logger.info(f"PCA completed: {X.shape} -> {X_pca.shape}")
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump((X_pca, pca, scaler), f)
        logger.debug(f"Saved PCA results to {cache_file}")

        pca_time = time.time() - start_time
        logger.info(f"PCA completed in {pca_time:.2f}s")
        
        return X_pca, pca, scaler
        
    except Exception as e:
        logger.error(f"PCA computation failed: {e}")
        raise ValueError(f"Failed to apply PCA: {e}")


def train_svm(
    X_train: np.ndarray, 
    y_train: List[int], 
    cache_dir: str = 'train_cache', 
    layer_name: str = 'conv3_block4_2_relu', 
    n_components: int = 500, 
    force_retrain: bool = False,
    pool_size: int = 2
) -> SVC:
    """Train SVM classifier on PCA-transformed features.
    
    Args:
        X_train (np.ndarray): Training features after PCA transformation, 
                             shape (n_samples, n_components)
        y_train (List[int]): Training labels as class indices
        cache_dir (str): Directory to store cached SVM model. 
                        Defaults to 'train_cache'
        layer_name (str): Name of the feature extraction layer for cache naming. 
                         Defaults to 'conv3_block4_2_relu'
        n_components (int): Number of PCA components used for cache naming. 
                           Defaults to 500
        force_retrain (bool): If True, ignore cached SVM and retrain. 
                             Defaults to False
        pool_size (int): Size of the max pooling layer used in feature extraction.
                        Defaults to 2. Used for cache file naming
    
    Returns:
        SVC: Trained Support Vector Machine classifier with probability estimates
        
    Raises:
        ValueError: If training data is invalid or SVM training fails
        OSError: If cache operations fail
    """
    if X_train.size == 0:
        raise ValueError("Training features array is empty")
    
    if len(y_train) == 0:
        raise ValueError("Training labels list is empty")
    
    if X_train.shape[0] != len(y_train):
        raise ValueError(f"Feature count ({X_train.shape[0]}) doesn't match label count ({len(y_train)})")
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = f"{cache_dir}/svm_pca_{n_components}_{layer_name}_pool{pool_size}.pkl"

    if os.path.exists(cache_file) and not force_retrain:
        logger.info(f"Loading cached SVM from {cache_file}")
        try:
            return pickle.load(open(cache_file, 'rb'))
        except Exception as e:
            logger.warning(f"Failed to load SVM cache, retraining: {e}")

    if force_retrain and os.path.exists(cache_file):
        logger.info(f"Force retrain: Removing cached SVM from {cache_file}")
        os.remove(cache_file)

    logger.info(f"Training SVM classifier on {X_train.shape[0]} samples with {X_train.shape[1]} features...")

    start_time = time.time()

    try:
        # Train SVM with linear kernel and probability estimates
        svm = SVC(
            kernel='linear', 
            probability=True,
            random_state=42,  # For reproducible results
            cache_size=1000   # Increase cache for better performance
        )
        
        svm.fit(X_train, y_train)
        
        # Log training results
        n_support = svm.n_support_.sum()
        logger.info(f"SVM training completed. Support vectors: {n_support}/{X_train.shape[0]}")
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(svm, f)
        logger.debug(f"Saved SVM model to {cache_file}")
        
        svm_time = time.time() - start_time
        logger.info(f"SVM training completed in {svm_time:.2f}s")
        
        return svm
        
    except Exception as e:
        logger.error(f"SVM training failed: {e}")
        raise ValueError(f"Failed to train SVM: {e}")


def predict_single_image(
    image_path: str, 
    svm: SVC, 
    pca: PCA, 
    scaler: StandardScaler, 
    feature_extractor: keras.Model, 
    class_mapping: Dict[str, int]
) -> Tuple[str, int]:
    """Predict elephant identity for a single image using the trained pipeline.
    
    Args:
        image_path (str): Path to the image file to classify
        svm (SVC): Trained SVM classifier
        pca (PCA): Fitted PCA transformer for dimensionality reduction
        scaler (StandardScaler): Fitted scaler for feature normalization
        feature_extractor (keras.Model): Pre-trained feature extraction model
        class_mapping (Dict[str, int]): Mapping from elephant names to class indices
        
    Returns:
        Tuple[str, int]: Tuple containing:
            - Predicted elephant name (or "UNKNOWN" if not found)
            - Predicted class index
            
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If prediction pipeline fails
    """
    try:
        # Extract features using GPU-optimized pipeline
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            image = load_image(image_path)
            feature_map = feature_extractor(image)
        
        # Convert to numpy and flatten
        raw_features = feature_map.numpy().flatten()

        # Apply preprocessing pipeline
        features_scaled = scaler.transform([raw_features])
        features_pca = pca.transform(features_scaled)

        # Make prediction
        prediction_id = svm.predict(features_pca)[0]

        # Find the name corresponding to the predicted ID
        matching_names = [name for name, id_val in class_mapping.items() if id_val == prediction_id]

        if not matching_names:
            prediction_name = "UNKNOWN"
            logger.warning(f"No name found for predicted class ID {prediction_id}")
        else:
            prediction_name = matching_names[0]

        return prediction_name, prediction_id
        
    except Exception as e:
        logger.error(f"Prediction failed for image {image_path}: {e}")
        raise ValueError(f"Failed to predict image {image_path}: {e}")


def evaluate_on_set(
    dataset: pd.DataFrame, 
    svm: SVC, 
    pca: PCA, 
    scaler: StandardScaler, 
    feature_extractor: keras.Model, 
    class_mapping: Dict[str, int],
    batch_size: Optional[int] = None,
    pool_size: int = 2
) -> float:
    """Evaluate trained model accuracy on a dataset using optimized batch processing.
    
    Args:
        dataset (pd.DataFrame): Dataset with columns ['filepath', 'name'] to evaluate on
        svm (SVC): Trained SVM classifier
        pca (PCA): Fitted PCA transformer
        scaler (StandardScaler): Fitted feature scaler
        feature_extractor (keras.Model): Pre-trained feature extraction model
        class_mapping (Dict[str, int]): Mapping from elephant names to class indices
        batch_size (Optional[int]): Batch size for feature extraction. If None, uses optimal size
        pool_size (int): Size of the max pooling layer used in feature extraction.
                        Defaults to 2. Must match the pool size used during training
        
    Returns:
        float: Accuracy score between 0.0 and 1.0
        
    Raises:
        ValueError: If dataset is empty or missing required columns
    """
    if dataset.empty:
        raise ValueError("Evaluation dataset is empty")
    
    required_columns = ['filepath', 'name']
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")
    
    try:
        # Use optimized batch feature extraction
        start_time = time.time()
        
        # Extract features for all images in batches
        raw_features, extracted_names = extract_raw_features(
            dataset, 
            feature_extractor, 
            layer_name='evaluation_temp',  # Use temp cache name
            cache_dir='temp_eval_cache', 
            force_retrain=True,  # Always recompute for evaluation
            batch_size=batch_size,
            pool_size=pool_size  # Use same pool size as training
        )
        
        feature_extraction_time = time.time() - start_time
        # already logged in extract_raw_features
        
        # Apply PCA transformation to all features at once
        start_time = time.time()
        features_scaled = scaler.transform(raw_features)
        features_pca = pca.transform(features_scaled)
        pca_time = time.time() - start_time
        logger.info(f"PCA completed in {pca_time:.2f}s")
        
        # Predict all samples at once
        start_time = time.time()
        predicted_ids = svm.predict(features_pca)
        prediction_time = time.time() - start_time
        logger.info(f"SVM prediction completed in {prediction_time:.2f}s")
        
        # Convert predicted IDs back to names
        reverse_class_mapping = {v: k for k, v in class_mapping.items()}
        predicted_names = [reverse_class_mapping.get(pred_id, "UNKNOWN") for pred_id in predicted_ids]
        
        # Compare predictions with true labels
        true_names = [str(name) for name in extracted_names]
        correct = sum(1 for true_name, pred_name in zip(true_names, predicted_names) 
                     if true_name == pred_name)
        
        # Calculate accuracy
        total = len(true_names)
        accuracy = correct / total if total > 0 else 0.0
        
        # Log detailed results
        total_time = feature_extraction_time + pca_time + prediction_time
        logger.debug(f"Batch evaluation completed in {total_time:.2f}s")
        logger.info(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
        
        # Clean up temporary cache
        temp_cache_dir = 'temp_eval_cache'
        if os.path.exists(temp_cache_dir):
            shutil.rmtree(temp_cache_dir)
            logger.debug("Cleaned up temporary evaluation cache")
        
        return accuracy
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        logger.info("Falling back to single-image evaluation...")
        
        # Fallback to single-image evaluation if batch fails
        return _evaluate_single_images(dataset, svm, pca, scaler, feature_extractor, class_mapping)


def _evaluate_single_images(
    dataset: pd.DataFrame,
    svm: SVC,
    pca: PCA,
    scaler: StandardScaler,
    feature_extractor: keras.Model,
    class_mapping: Dict[str, int]
) -> float:
    """Fallback single-image evaluation method.
    
    Args:
        dataset (pd.DataFrame): Dataset to evaluate
        svm (SVC): Trained SVM classifier
        pca (PCA): Fitted PCA transformer
        scaler (StandardScaler): Fitted feature scaler
        feature_extractor (keras.Model): Pre-trained feature extraction model
        class_mapping (Dict[str, int]): Mapping from elephant names to class indices
        
    Returns:
        float: Accuracy score between 0.0 and 1.0
    """
    correct = 0
    total = len(dataset)
    errors = 0

    logger.info(f"Fallback evaluation on {total} images (single-image processing)...")
    
    for idx, row in dataset.iterrows():
        try:
            predicted_name, _ = predict_single_image(
                row['filepath'], svm, pca, scaler, feature_extractor, class_mapping
            )
            if predicted_name == str(row['name']):
                correct += 1
        except Exception as e:
            errors += 1
            logger.warning(f"Evaluation error for image {row['filepath']}: {e}")
        
        # Progress logging
        if (idx + 1) % 50 == 0:
            logger.info(f"Evaluated {idx + 1}/{total} images")

    if errors > 0:
        logger.warning(f"Encountered {errors} errors during evaluation")
    
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Single-image evaluation complete - Accuracy: {accuracy:.3f} ({correct}/{total})")
    return accuracy


def create_optimized_feature_extractor(
    layer_name: str = 'conv3_block4_2_relu',
    pool_size: int = 2
) -> keras.Model:
    """Create an optimized feature extractor for Apple Silicon.
    
    Args:
        layer_name (str): Name of the ResNet50 layer to extract features from.
                         Defaults to 'conv3_block4_2_relu'
        pool_size (int): Size of the max pooling layer. Defaults to 2
                         
    Returns:
        keras.Model: Optimized feature extraction model
        
    Raises:
        ValueError: If the specified layer name doesn't exist in ResNet50
    """
    try:
        # Create ResNet50 base model
        model = ResNet50(
            include_top=False, 
            weights='imagenet', 
            input_shape=(224, 224, 3)
        )
        
        # Get target layer
        target_layer = model.get_layer(name=layer_name)
        
        if pool_size > 1:
            # Add max pooling for dimensionality reduction
            max_pooling = keras.layers.MaxPooling2D(
                pool_size=(pool_size, pool_size), 
                name='feature_pooling'
            )(target_layer.output)
        
            # Create feature extractor model
            feature_extractor = keras.Model(
                inputs=model.input, 
                outputs=max_pooling,
                name='optimized_feature_extractor'
            )
        else:
            feature_extractor = keras.Model(
                inputs=model.input, 
                outputs=target_layer.output,
                name='optimized_feature_extractor'
            )
        
        logger.debug(f"Created feature extractor from layer: {layer_name}")
        logger.debug(f"Output shape: {feature_extractor.output_shape}")
        
        return feature_extractor
        
    except ValueError as e:
        available_layers = [layer.name for layer in model.layers]
        logger.error(f"Layer '{layer_name}' not found. Available layers: {available_layers[:10]}...")
        raise ValueError(f"Invalid layer name '{layer_name}': {e}")


if __name__ == "__main__":
    # Configure Apple Silicon optimizations
    configure_apple_silicon_gpu()
    
    parser = argparse.ArgumentParser(
        description='Train and evaluate elephant identification model with Apple Silicon optimizations'
    )
    parser.add_argument(
        '--force', 
        action='store_true', 
        help='Force retrain by clearing all cached files'
    )
    parser.add_argument(
        '--force-pca', 
        action='store_true', 
        help='Force retrain by clearing all cached PCA files'
    )
    parser.add_argument(
        '--force-svm', 
        action='store_true', 
        help='Force retrain by clearing all cached SVM files'
    )
    parser.add_argument(
        '--n-components', 
        type=int, 
        default=350, 
        help='Number of PCA components (default: 350)'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        help='Batch size for feature extraction (auto-detected if not specified)'
    )
    parser.add_argument(
        '--layer-name', 
        type=str, 
        default='conv3_block4_2_relu',
        help='ResNet50 layer name for feature extraction (default: conv3_block4_2_relu)'
    )
    parser.add_argument(
        '--pool-size', 
        type=int, 
        default=2, 
        help='Size of the max pooling layer (default: 2)'
    )
    args = parser.parse_args()

    logger.info("Starting elephant identification pipeline...")
    
    # Log configuration
    gpu_available = bool(tf.config.list_physical_devices('GPU'))
    logger.debug(f"Apple Silicon GPU available: {gpu_available}")
    logger.debug(f"TensorFlow version: {tf.__version__}")
    
    if args.force:
        logger.info("Force retrain: All cached files will be cleared and retrained")
    if args.force_pca:
        logger.info("Force retrain: All cached PCA files will be cleared and retrained")
    if args.force_svm:
        logger.info("Force retrain: All cached SVM files will be cleared and retrained")
    
    if args.batch_size:
        logger.info(f"Using batch size: {args.batch_size}")

    try:
        # Load training data
        root_dir = "/Users/kayoko/Documents/GitHub/elephant-identification"
        train_data = pd.read_csv(f"{root_dir}/dataset/train.csv")
        logger.info(f"Loaded {len(train_data)} training samples")

        # Load class mapping
        with open('dataset/class_mapping.json', 'r') as f:
            class_mapping = json.load(f)
        logger.info(f"Loaded class mapping for {len(class_mapping)} elephants")

        layer_name = args.layer_name
        n_components = args.n_components

        logger.info(f"Training pipeline: ResNet50 until {layer_name} -> Pool({args.pool_size}) -> PCA({n_components}) -> SVM")

        # Create optimized feature extractor
        feature_extractor = create_optimized_feature_extractor(layer_name, args.pool_size)

        # Extract raw features with GPU optimization
        raw_features, names = extract_raw_features(
            train_data, 
            feature_extractor, 
            layer_name, 
            force_retrain=args.force,
            batch_size=args.batch_size,
            pool_size=args.pool_size
        )
        logger.debug(f"Raw features shape: {np.array(raw_features).shape}")

        # Apply PCA with enhanced error handling
        X_pca, pca, scaler = apply_pca(
            raw_features, 
            n_components, 
            layer_name=layer_name, 
            force_retrain=args.force_pca or args.force,
            pool_size=args.pool_size
        )
        logger.debug(f"PCA features shape: {X_pca.shape}")

        # Convert names to class IDs
        y_train = [class_mapping[name] for name in names]
        logger.debug(f"Converted {len(y_train)} labels to class IDs")

        # Train SVM with enhanced configuration
        svm = train_svm(
            X_pca, 
            y_train, 
            layer_name=layer_name, 
            n_components=n_components, 
            force_retrain=args.force_svm or args.force,
            pool_size=args.pool_size
        )

        # Load test data and evaluate
        test_data = pd.read_csv(f"{root_dir}/dataset/test.csv")
        logger.info(f"Loaded {len(test_data)} test samples")

        # Evaluate on test set with batch optimization
        logger.info("Starting test set evaluation...")
        evaluate_on_set(
            test_data, svm, pca, scaler, feature_extractor, class_mapping,
            batch_size=args.batch_size, pool_size=args.pool_size
        )
        
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise



# Performance notes:
# - Baseline accuracy: 0.268 with non-reflected images
# - With reflection augmentation: 0.253 
# - Apple Silicon GPU optimization provides 2-4x speedup for feature extraction
# - Mixed precision training reduces memory usage by ~40%