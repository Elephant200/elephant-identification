import argparse
import json
import logging
import os
import pickle
import shutil
import time
from typing import Dict, List, Literal, Optional, Tuple

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from tqdm import tqdm

from utils import print_with_padding

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def configure_tensorflow(device: Literal['CPU', 'CUDA', 'MPS']) -> None:
    """Configure TensorFlow for optimal usage.

    Devices:
        - CPU: CPU backend
        - CUDA: CUDA backend
        - MPS: Metal Performance Shaders backend
    
    Sets up the backend, configures memory growth, and optimizes TensorFlow settings.
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
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
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
        image_raw = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_raw, channels=3)
        
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32)
        
        image = tf.expand_dims(image, 0)
        image = keras.applications.resnet.preprocess_input(image)
        
        return image
        
    except tf.errors.NotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise ValueError(f"Failed to load and preprocess image {image_path}: {e}")


def extract_raw_features(
    data_df: pd.DataFrame, 
    feature_extractor: keras.Model, 
    layer_name: str, 
    cache_dir: str = 'identify/train_cache', 
    force_retrain: bool = False,
    batch_size: int | None = None,
    pool_size: int = 2
) -> list[np.ndarray]:
    """Extract and cache raw features from images
    
    Args:
        data_df (pd.DataFrame): DataFrame containing image paths and labels with columns 
                               ['filepath', 'name']
        feature_extractor (keras.Model): Pre-trained model for feature extraction
        layer_name (str): Name of the layer to extract features from. Only used for cache file naming
                         Defaults to 'conv3_block4_2_relu'
        cache_dir (str): Directory to store cached features. Defaults to 'identify/train_cache'
        force_retrain (bool): If True, ignore cached features and recompute. 
                             Defaults to False
        batch_size (int | None): Batch size for processing. If None, uses optimal size
                                   based on hardware detection
        pool_size (int): Size of the max pooling layer used in feature extraction. Only used for cache file naming.
                        Defaults to 2.
    
    Returns:
        list[np.ndarray]: List of flattened feature arrays for each image
    """
    # Validate inputs
    if data_df.empty:
        raise ValueError("Input DataFrame is empty")
    
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
        
    # Process images in batches for better GPU utilization
    for i in tqdm(range(0, len(data_df), batch_size), desc=f"Extracting raw features with batch size {batch_size}: "):
        batch_df = data_df.iloc[i:i + batch_size]
        batch_images = []
        
        # Load batch of images
        for _, row in batch_df.iterrows():
            try:
                image = load_image(row['filepath'])
                batch_images.append(image)
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
            for feature_map in batch_features.numpy():
                features.append(feature_map.flatten())
        
    logger.debug(f"Saving {len(features)} features to {cache_file}")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(features, f)
    except Exception as e:
        logger.error(f"Failed to save features cache: {e}")
        raise OSError(f"Could not save features to {cache_file}: {e}")
        
    return features


def train_pca(
    features: List[np.ndarray], 
    n_components: int = 500, 
    cache_dir: str = 'identify/train_cache', 
    layer_name: str = 'conv3_block4_2_relu', 
    force_retrain: bool = False,
    pool_size: int = 2
) -> Tuple[np.ndarray, PCA, StandardScaler]:
    """Train PCA dimensionality reduction to extracted features and return fitted PCA and scaler.
    
    Args:
        features (List[np.ndarray]): List of flattened feature vectors from images
        n_components (int): Number of principal components to retain. 
                           Defaults to 500
        cache_dir (str): Directory to store cached PCA results. 
                        Defaults to 'identify/train_cache'
        layer_name (str): Name of the feature extraction layer for cache naming. 
                         Defaults to 'conv3_block4_2_relu'
        force_retrain (bool): If True, ignore cached PCA and recompute. 
                             Defaults to False
        pool_size (int): Size of the max pooling layer used in feature extraction.
                        Defaults to 2. Used for cache file naming
    
    Returns:
        Tuple[np.ndarray, PCA, StandardScaler]: Tuple containing:
            - Transformed features with reduced dimensionality (n_samples, n_components)
            - Fitted PCA transformer
            - Fitted StandardScaler
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

    logger.info(f"Training PCA dimensionality reduction to {len(features)} features...")
    
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
                
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump((X_pca, pca, scaler), f)
        logger.debug(f"Saved PCA results to {cache_file}")
        
        return X_pca, pca, scaler
        
    except Exception as e:
        logger.error(f"PCA computation failed: {e}")
        raise ValueError(f"Failed to train PCA: {e}")


def train_svm(
    X_train: np.ndarray, 
    y_train: List[int], 
    cache_dir: str = 'identify/train_cache', 
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
                        Defaults to 'identify/train_cache'
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


def train_on_set(
    dataset: pd.DataFrame,
    layer_name: str,
    n_components: int,
    pool_size: int,
    class_mapping: Dict[str, int],
    *,
    force_features: bool = False,
    force_pca: bool = False,
    force_svm: bool = False,
    batch_size: int | None = None,
):
    """Train the pipeline on the train dataset"""
    try:
        logger.info(f"Training pipeline: ResNet50 until {layer_name} -> Pool({pool_size}) -> PCA({n_components}) -> SVM")

        # Create optimized feature extractor
        feature_extractor = create_feature_extractor(layer_name, pool_size)

        # Extract raw features with GPU optimization
        start_time = time.perf_counter()
        raw_features = extract_raw_features(
            dataset, 
            feature_extractor, 
            layer_name, 
            force_retrain=force_features,
            batch_size=batch_size,
            pool_size=pool_size
        )
        feature_extraction_time = time.perf_counter() - start_time
        logger.info(f"Feature extraction completed in {feature_extraction_time:.2f}s")
        logger.debug(f"Raw features shape: {np.array(raw_features).shape}")

        # Train PCA
        start_time = time.perf_counter()
        X_pca, pca, scaler = train_pca(
            raw_features, 
            n_components, 
            layer_name=layer_name, 
            force_retrain=force_pca,
            pool_size=pool_size
        )
        pca_time = time.perf_counter() - start_time
        logger.info(f"PCA training completed in {pca_time:.2f}s")
        logger.debug(f"PCA features shape: {X_pca.shape}")

        # Convert names to class IDs
        y_train = [class_mapping[str(name)] for name in dataset['name']]
        logger.debug(f"Converted {len(y_train)} labels to class IDs")

        # Train SVM
        start_time = time.perf_counter()
        svm = train_svm(
            X_pca, 
            y_train, 
            layer_name=layer_name, 
            n_components=n_components, 
            force_retrain=force_svm,
            pool_size=pool_size
        )
        svm_time = time.perf_counter() - start_time
        logger.info(f"SVM training completed in {svm_time:.2f}s")

        return feature_extractor, pca, scaler, svm

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise ValueError(f"Failed to train on set: {e}")


def evaluate_on_set(
    dataset: pd.DataFrame, 
    svm: SVC, 
    pca: PCA, 
    scaler: StandardScaler, 
    feature_extractor: keras.Model, 
    class_mapping: Dict[str, int],
    layer_name: str,
    batch_size: int | None = None,
    pool_size: int = 2,
    top_k_values: List[int] = [1, 3, 5, 10],
    force: bool = False
) -> dict[str, dict[str, float]]:
    """Evaluate trained model accuracy on a dataset using optimized batch processing.
    
    Args:
        dataset (pd.DataFrame): Dataset with columns ['filepath', 'name'] to evaluate on
        svm (SVC): Trained SVM classifier
        pca (PCA): Fitted PCA transformer
        scaler (StandardScaler): Fitted feature scaler
        feature_extractor (keras.Model): Pre-trained feature extraction model
        class_mapping (Dict[str, int]): Mapping from elephant names to class indices
        layer_name (str): Name of the layer to extract features from.
        batch_size (Optional[int]): Batch size for feature extraction. If None, uses optimal size
        pool_size (int): Size of the max pooling layer used in feature extraction.
                        Defaults to 2. Must match the pool size used during training
        top_k_values (List[int]): List of k values for top-k accuracy calculation.
                                 Defaults to [1, 3, 5]
        
    Returns:
        dict[str, dict[str, float]]: Dictionary mapping 'top_k' to accuracy percentage, number of correct predictions, and number of total predictions
        
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
        # Extract features for all images in batches
        start_time = time.perf_counter()
        raw_features = extract_raw_features(
            dataset, 
            feature_extractor, 
            layer_name=layer_name,
            cache_dir='identify/test_cache', 
            force_retrain=force,
            batch_size=batch_size,
            pool_size=pool_size
        )
        feature_extraction_time = time.perf_counter() - start_time
        logger.info(f"Feature extraction completed in {feature_extraction_time:.2f}s")
        
        # Apply PCA transformation to all features at once
        start_time = time.perf_counter()
        features_scaled = scaler.transform(raw_features)
        features_pca = pca.transform(features_scaled)
        pca_time = time.perf_counter() - start_time
        logger.info(f"PCA completed in {pca_time:.2f}s")
        
        # Get prediction probabilities for all classes
        start_time = time.perf_counter()
        prediction_probs = svm.predict_proba(features_pca)
        predictions = svm.predict(features_pca)
        prediction_time = time.perf_counter() - start_time
        logger.info(f"SVM prediction completed in {prediction_time:.2f}s")
        
        # Convert true names to IDs for comparison
        true_names = [str(name) for name in dataset['name']]
        true_ids = [class_mapping.get(name, -1) for name in true_names]

        # Calculate accuracy using sklearn metrics
        accuracy = accuracy_score(true_ids, predictions)
        correct = int(accuracy * len(true_ids))
        logger.info(f"Direct prediction accuracy: {accuracy:.2%} ({correct}/{len(true_ids)})")

        top_1_accuracy = top_k_accuracy_score(true_ids, prediction_probs, k=1)
        top_3_accuracy = top_k_accuracy_score(true_ids, prediction_probs, k=3)
        top_5_accuracy = top_k_accuracy_score(true_ids, prediction_probs, k=5)
        top_10_accuracy = top_k_accuracy_score(true_ids, prediction_probs, k=10)

        # Print out probabilities for each of the 10 highest probability classes, ordered by probability
        top10_probs = np.sort(prediction_probs, axis=1)[:, ::-1][:, :10]
        for i in range(10):
            print(f"Top {i+1} probability: {top10_probs[0][i]}")

        
        
        accuracies = {
            "top_1": {"accuracy": top_1_accuracy, "correct": round(len(true_ids) * top_1_accuracy), "total": len(true_ids)},
            "top_3": {"accuracy": top_3_accuracy, "correct": round(len(true_ids) * top_3_accuracy), "total": len(true_ids)},
            "top_5": {"accuracy": top_5_accuracy, "correct": round(len(true_ids) * top_5_accuracy), "total": len(true_ids)},
            "top_10": {"accuracy": top_10_accuracy, "correct": round(len(true_ids) * top_10_accuracy), "total": len(true_ids)}
        }
        
        # Log detailed results
        total_time = feature_extraction_time + pca_time + prediction_time
        logger.debug(f"Evaluation completed in {total_time:.2f}s")
        
        # Clean up temporary cache
        temp_cache_dir = 'identify/temp_eval_cache'
        if os.path.exists(temp_cache_dir):
            shutil.rmtree(temp_cache_dir)
            logger.debug("Cleaned up temporary evaluation cache")
        
        return accuracies
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


def create_feature_extractor(
    layer_name: str,
    pool_size: int = 6
) -> keras.Model:
    """Create a feature extractor.
    
    Args:
        layer_name (str): Name of the ResNet50 layer to extract features from.
        pool_size (int): Size of the max pooling layer. Defaults to 6
                         
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
            max_pooling_layer = keras.layers.MaxPooling2D(
                pool_size=(pool_size, pool_size), 
                name='feature_pooling'
            )(target_layer.output)

            last_layer = max_pooling_layer
        else:
            last_layer = target_layer.output

        feature_extractor = keras.Model(
                inputs=model.input, 
                outputs=last_layer,
                name='feature_extractor'
            )
        
        logger.debug(f"Created feature extractor from layer: {layer_name}")
        logger.debug(f"Output shape: {feature_extractor.output_shape}")
        
        return feature_extractor
        
    except ValueError as e:
        logger.error(f"Layer '{layer_name}' not found.")
        raise ValueError(f"Invalid layer name '{layer_name}': {e}")

def run_and_evaluate(
        force_features: bool, 
        force_pca: bool, 
        force_svm: bool, 
        batch_size: int | None, 
        layer_name: str, 
        n_components: int, 
        pool_size: int,
        top_k_values: List[int]) -> None:
    """Run the pipeline and evaluate the model"""
    configure_tensorflow(device='MPS')
    
    # Load training data
    try:
        root_dir = "/Users/kayoko/Documents/GitHub/elephant-identification"
        train_data = pd.read_csv(f"{root_dir}/dataset/appearance_metadata/train.csv")
        logger.info(f"Loaded {len(train_data)} training samples")

        # Load class mapping
        with open('dataset/appearance_metadata/class_mapping.json', 'r') as f:
            class_mapping = json.load(f)
        logger.info(f"Loaded class mapping for {len(class_mapping)} elephants")

        feature_extractor, pca, scaler, svm = train_on_set(
            dataset=train_data, 
            layer_name=layer_name, 
            n_components=n_components, 
            pool_size=pool_size,
            class_mapping=class_mapping,
            force_features=force_features,
            force_pca=force_pca,
            force_svm=force_svm,
            batch_size=batch_size
        )

        # Load test data and evaluate
        test_data = pd.read_csv(f"{root_dir}/dataset/appearance_metadata/test.csv")
        logger.info(f"Loaded {len(test_data)} test samples")

        # Evaluate on test set with batch optimization
        logger.info("Starting test set evaluation...")
        accuracies = evaluate_on_set(
            dataset=test_data, 
            svm=svm, 
            pca=pca, 
            scaler=scaler, 
            feature_extractor=feature_extractor, 
            class_mapping=class_mapping,
            batch_size=batch_size, 
            layer_name=layer_name,
            pool_size=pool_size, 
            top_k_values=top_k_values,
            force=force_features
        )
        for k, v in accuracies.items():
            logger.info(f"{k}: Accuracy: {v['accuracy']:.3f} ({v['correct']}/{v['total']})")

        return accuracies

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    configure_tensorflow(device='MPS')
    
    parser = argparse.ArgumentParser(
        description='Train and evaluate elephant identification model with Apple Silicon optimizations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        default=10000, 
        help='Number of PCA components (default: 10000)'
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
        default=6, 
        help='Size of the max pooling layer (default: 6)'
    )
    parser.add_argument(
        '--top-k', 
        type=int, 
        nargs='+',
        default=[1, 3, 5, 10], 
        help='List of k values for top-k accuracy evaluation (default: 1 3 5 10)'
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

    force_features: bool = args.force
    force_pca: bool = args.force_pca or args.force
    force_svm: bool = args.force_svm or args.force

    layer_name: str = args.layer_name
    n_components: int = args.n_components
    pool_size: int = args.pool_size
    batch_size: int | None = args.batch_size

    run_and_evaluate(
        force_features=force_features, 
        force_pca=force_pca, 
        force_svm=force_svm, 
        batch_size=batch_size, 
        layer_name=layer_name, 
        n_components=n_components, 
        pool_size=pool_size,
        top_k_values=args.top_k
    )

# Performance notes:
# - Baseline accuracy: 0.268 with non-reflected images
# - With reflection augmentation: 0.253 
# - Apple Silicon GPU optimization provides 2-4x speedup for feature extraction
# - Mixed precision training reduces memory usage by ~40%