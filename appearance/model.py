"""Elephant identification model classes.

Provides a base ElephantIdentifier class with PCA + SVM classification,
and subclasses for different feature extractors (ResNet50, MegaDescriptor).
"""
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from .core import get_device, get_optimal_batch_size

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = 'cache/appearance/features'


class ElephantIdentifier(ABC):
    """Abstract base class for elephant identification models.

    Combines feature extraction with StandardScaler, PCA dimensionality
    reduction, and SVM classification. Subclasses implement the feature
    extraction logic for specific model architectures.

    Attributes:
        n_components: Number of PCA components
        class_mapping: Dict mapping elephant name/ID to class index
        reverse_mapping: Dict mapping class index to elephant name/ID
    """

    # Subclasses should override this
    MODEL_TYPE: str = "base"

    def __init__(self, n_components: int = 1024):
        """Initialize the identifier with model hyperparameters.

        Args:
            n_components: Number of PCA components to retain
        """
        self.n_components = n_components

        self._feature_extractor: nn.Module | None = None
        self._transform: Callable[[Image.Image], torch.Tensor] | None = None
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._svm: SVC | None = None
        self._class_mapping: Dict[str, int] | None = None
        self._reverse_mapping: Dict[int, str] | None = None
        self._is_fitted = False

    @property
    def class_mapping(self) -> Dict[str, int] | None:
        return self._class_mapping

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def _create_feature_extractor(self) -> nn.Module:
        """Create the feature extractor model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_transform(self) -> Callable[[Image.Image], torch.Tensor]:
        """Get the image transform for this model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_cache_filename(self) -> str:
        """Get the cache filename for this model configuration."""
        pass

    @abstractmethod
    def _get_model_state(self) -> Dict:
        """Get model-specific state for saving. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_training_description(self) -> str:
        """Get a description of the training pipeline for logging."""
        pass

    @abstractmethod
    def _should_flatten_features(self) -> bool:
        """Whether to flatten spatial feature maps to 1D vectors."""
        pass

    def _ensure_feature_extractor(self) -> nn.Module:
        """Lazily create the feature extractor."""
        if self._feature_extractor is None:
            self._feature_extractor = self._create_feature_extractor()
            self._feature_extractor.eval()
            self._feature_extractor = self._feature_extractor.to(get_device())
        return self._feature_extractor

    def _ensure_transform(self) -> Callable[[Image.Image], torch.Tensor]:
        """Lazily create the image transform."""
        if self._transform is None:
            self._transform = self._get_transform()
        return self._transform

    def _get_cache_path(self, cache_dir: str) -> str:
        """Get cache file path for features."""
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, self._get_cache_filename())

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess an image.

        Args:
            image_path: Path to the image file

        Returns:
            torch.Tensor: Preprocessed image tensor with batch dimension

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the image cannot be loaded or processed
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        transform = self._ensure_transform()
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = transform(image)
            return tensor.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            raise ValueError(f"Failed to load and preprocess image {image_path}: {e}")

    def _extract_features_batch(
        self,
        data_df: pd.DataFrame,
        batch_size: int | None = None,
        cache_path: str | None = None,
        force: bool = False
    ) -> List[np.ndarray]:
        """Extract features from images in batches.

        Args:
            data_df: DataFrame with 'filepath' column containing image paths
            batch_size: Batch size for processing. If None, uses optimal size.
            cache_path: Optional path to cache extracted features
            force: If True, ignore cached features and recompute

        Returns:
            List[np.ndarray]: List of feature arrays for each image
        """
        if data_df.empty:
            raise ValueError("Input DataFrame is empty")

        if cache_path and os.path.exists(cache_path) and not force:
            logger.info(f"Loading cached features from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache, recomputing: {e}")

        if batch_size is None:
            batch_size = get_optimal_batch_size()

        features: List[np.ndarray] = []
        device = get_device()
        feature_extractor = self._ensure_feature_extractor()
        flatten = self._should_flatten_features()

        for i in tqdm(range(0, len(data_df), batch_size), desc=f"Extracting features (batch={batch_size})"):
            batch_df = data_df.iloc[i:i + batch_size]
            batch_images = []

            for _, row in batch_df.iterrows():
                try:
                    image = self._load_image(row['filepath'])
                    batch_images.append(image)
                except Exception as e:
                    logger.warning(f"Failed to load image {row['filepath']}: {e}")
                    continue

            if batch_images:
                batch_tensor = torch.cat(batch_images, dim=0).to(device)

                with torch.no_grad():
                    batch_features = feature_extractor(batch_tensor)

                # Handle dict output (from torchvision feature_extraction)
                if isinstance(batch_features, dict):
                    batch_features = list(batch_features.values())[0]

                for feature_vec in batch_features.cpu().numpy():
                    if flatten and feature_vec.ndim > 1:
                        features.append(feature_vec.flatten())
                    else:
                        features.append(feature_vec)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            logger.debug(f"Saving {len(features)} features to {cache_path}")
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(features, f)
            except Exception as e:
                logger.error(f"Failed to save features cache: {e}")

        return features

    def _extract_single_image_features(self, image_path: str) -> np.ndarray:
        """Extract features from a single image.

        Args:
            image_path: Path to the image file

        Returns:
            np.ndarray: Feature array
        """
        device = get_device()
        feature_extractor = self._ensure_feature_extractor()
        flatten = self._should_flatten_features()

        image = self._load_image(image_path).to(device)

        with torch.no_grad():
            feature_vec = feature_extractor(image)

        # Handle dict output (from torchvision feature_extraction)
        if isinstance(feature_vec, dict):
            feature_vec = list(feature_vec.values())[0]

        result = feature_vec.cpu().numpy().squeeze()

        if flatten and result.ndim > 1:
            return result.flatten()
        return result

    def fit(
        self,
        train_df: pd.DataFrame,
        class_mapping: Dict[str, int],
        batch_size: int | None = None,
        cache_dir: str = DEFAULT_CACHE_DIR,
        force: bool = False
    ) -> 'ElephantIdentifier':
        """Train the model on a dataset.

        Args:
            train_df: DataFrame with columns ['filepath', 'name']
            class_mapping: Dict mapping elephant name/ID to class index
            batch_size: Batch size for feature extraction. Auto-detected if None.
            cache_dir: Directory to cache intermediate results.
            force: If True, ignore cached data and retrain from scratch.

        Returns:
            self: The fitted model
        """
        logger.info(f"Training pipeline: {self._get_training_description()}")

        self._class_mapping = class_mapping
        self._reverse_mapping = {v: k for k, v in class_mapping.items()}

        cache_path = self._get_cache_path(cache_dir)

        start_time = time.perf_counter()
        raw_features = self._extract_features_batch(
            train_df,
            batch_size=batch_size,
            cache_path=cache_path,
            force=force
        )
        logger.info(f"Feature extraction completed in {time.perf_counter() - start_time:.2f}s")

        start_time = time.perf_counter()
        X = np.array(raw_features)

        if X.shape[0] != len(train_df):
            raise ValueError(
                f"Feature cache mismatch: cached features have {X.shape[0]} samples "
                f"but training data has {len(train_df)} samples. "
                f"The cache may be stale. Rerun with --force to recompute features."
            )

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        actual_components = min(self.n_components, X_scaled.shape[0], X_scaled.shape[1])
        if actual_components != self.n_components:
            logger.warning(f"Requested {self.n_components} PCA components, "
                           f"using {actual_components} based on data shape")

        self._pca = PCA(n_components=actual_components)
        X_pca = self._pca.fit_transform(X_scaled)
        logger.info(f"PCA training completed in {time.perf_counter() - start_time:.2f}s")
        logger.debug(f"PCA features shape: {X_pca.shape}")

        y_train = [class_mapping[str(name)] for name in train_df['name']]

        start_time = time.perf_counter()
        self._svm = SVC(
            kernel='linear',
            probability=True,
            random_state=42,
            cache_size=1000
        )
        self._svm.fit(X_pca, y_train)
        n_support = self._svm.n_support_.sum()
        logger.info(f"SVM training completed in {time.perf_counter() - start_time:.2f}s")
        logger.info(f"Support vectors: {n_support}/{X_pca.shape[0]}")

        self._is_fitted = True
        return self

    def predict(self, image_path: str) -> List[Tuple[str, float]]:
        """Predict elephant identity for a single image.

        Args:
            image_path: Path to the image file

        Returns:
            List of (name, confidence) tuples for all classes, sorted by confidence descending

        Raises:
            RuntimeError: If model has not been fitted
            FileNotFoundError: If image file doesn't exist
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        raw_features = self._extract_single_image_features(image_path)

        features_scaled = self._scaler.transform([raw_features])
        features_pca = self._pca.transform(features_scaled)

        probabilities = self._svm.predict_proba(features_pca)[0]
        class_indices = self._svm.classes_

        results = []
        for class_idx, prob in zip(class_indices, probabilities):
            name = self._reverse_mapping.get(class_idx, "UNKNOWN")
            results.append((name, float(prob)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int | None = None,
        cache_dir: str = DEFAULT_CACHE_DIR,
        force: bool = False
    ) -> List[List[Tuple[str, float]]]:
        """Predict elephant identities for multiple images.

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for feature extraction
            cache_dir: Directory to cache features
            force: If True, ignore cached features

        Returns:
            List of prediction lists. Each prediction list contains 
            (name, confidence) tuples for all classes, sorted by confidence descending.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        df = pd.DataFrame({'filepath': image_paths})
        cache_path = self._get_cache_path(cache_dir)

        raw_features = self._extract_features_batch(
            df,
            batch_size=batch_size,
            cache_path=cache_path,
            force=force
        )

        features_scaled = self._scaler.transform(raw_features)
        features_pca = self._pca.transform(features_scaled)

        probabilities = self._svm.predict_proba(features_pca)
        class_indices = self._svm.classes_

        results = []
        for probs in probabilities:
            image_results = []
            for class_idx, prob in zip(class_indices, probs):
                name = self._reverse_mapping.get(class_idx, "UNKNOWN")
                image_results.append((name, float(prob)))
            image_results.sort(key=lambda x: x[1], reverse=True)
            results.append(image_results)

        return results

    def evaluate(
        self,
        test_df: pd.DataFrame,
        batch_size: int | None = None,
        top_k_values: List[int] | None = None,
        cache_dir: str = DEFAULT_CACHE_DIR,
        force: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model accuracy on a test dataset.

        Args:
            test_df: DataFrame with columns ['filepath', 'name']
            batch_size: Batch size for feature extraction
            top_k_values: List of k values for top-k accuracy. Defaults to [1, 3, 5, 10]
            cache_dir: Directory to cache test features
            force: If True, ignore cached features

        Returns:
            Dict mapping 'top_k' to accuracy metrics
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if top_k_values is None:
            top_k_values = [1, 3, 5, 10]

        cache_path = self._get_cache_path(cache_dir)

        start_time = time.perf_counter()
        raw_features = self._extract_features_batch(
            test_df,
            batch_size=batch_size,
            cache_path=cache_path,
            force=force
        )
        logger.info(f"Feature extraction completed in {time.perf_counter() - start_time:.2f}s")

        features_scaled = self._scaler.transform(raw_features)
        features_pca = self._pca.transform(features_scaled)

        predictions = self._svm.predict(features_pca)
        prediction_probs = self._svm.predict_proba(features_pca)

        true_names = [str(name) for name in test_df['name']]
        true_ids = [self._class_mapping.get(name, -1) for name in true_names]

        accuracies = {}
        for k in top_k_values:
            if k > prediction_probs.shape[1]:
                logger.warning(f"Skipping top-{k} (only {prediction_probs.shape[1]} classes)")
                continue
            acc = top_k_accuracy_score(true_ids, prediction_probs, k=k)
            accuracies[f"top_{k}"] = {
                "accuracy": acc,
                "correct": round(len(true_ids) * acc),
                "total": len(true_ids)
            }

        return accuracies

    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path: Path to save the model pickle file

        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model. Call fit() first.")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        state = {
            'model_type': self.MODEL_TYPE,
            'n_components': self.n_components,
            'scaler': self._scaler,
            'pca': self._pca,
            'svm': self._svm,
            'class_mapping': self._class_mapping,
            'reverse_mapping': self._reverse_mapping,
        }
        # Add model-specific state
        state.update(self._get_model_state())

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'ElephantIdentifier':
        """Load a trained model from disk.

        This is a polymorphic loader that returns the appropriate subclass
        based on the saved model_type.

        Args:
            path: Path to the saved model pickle file

        Returns:
            ElephantIdentifier: Loaded model ready for inference (correct subclass)
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        model_type = state.get('model_type', 'resnet50')

        if model_type == 'resnet50':
            model = ResNet50Identifier(
                layer_name=state.get('layer_name', 'layer3'),
                pool_size=state.get('pool_size', 6),
                n_components=state['n_components']
            )
        elif model_type == 'megadescriptor':
            model = MegaDescriptorIdentifier(n_components=state['n_components'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model._scaler = state['scaler']
        model._pca = state['pca']
        model._svm = state['svm']
        model._class_mapping = state['class_mapping']
        model._reverse_mapping = state['reverse_mapping']
        model._is_fitted = True

        logger.info(f"Model loaded from {path}")
        return model

    @abstractmethod
    def __repr__(self) -> str:
        pass


class ResNet50Identifier(ElephantIdentifier):
    """Elephant identifier using ResNet50 feature extraction.

    Extracts features from an intermediate ResNet50 layer, applies optional
    max pooling, then uses PCA + SVM for classification.

    Layer options and their output shapes (with 224x224 input):
        - layer1: 56x56x256
        - layer2: 28x28x512
        - layer3: 14x14x1024
        - layer4: 7x7x2048

    Attributes:
        layer_name: ResNet50 layer name (layer1, layer2, layer3, layer4)
        pool_size: Max pooling size after feature extraction
    """

    MODEL_TYPE = "resnet50"
    INPUT_SIZE = 224

    # ResNet50 layer spatial dimensions (with 224x224 input)
    LAYER_SPATIAL_SIZES = {
        'layer1': 56,
        'layer2': 28,
        'layer3': 14,
        'layer4': 7,
    }

    def __init__(
        self,
        layer_name: str = 'layer3',
        pool_size: int = 6,
        n_components: int = 10000
    ):
        """Initialize ResNet50 identifier.

        Args:
            layer_name: ResNet50 layer name (layer1, layer2, layer3, layer4)
            pool_size: Max pooling size. Use 1 to disable pooling.
            n_components: Number of PCA components to retain
        """
        super().__init__(n_components=n_components)

        if layer_name not in self.LAYER_SPATIAL_SIZES:
            raise ValueError(
                f"Invalid layer_name '{layer_name}'. "
                f"Must be one of: {list(self.LAYER_SPATIAL_SIZES.keys())}"
            )

        self.layer_name = layer_name
        self.pool_size = pool_size

    def _create_feature_extractor(self) -> nn.Module:
        """Create ResNet50 feature extractor."""
        logger.info(f"Loading ResNet50 (extracting from {self.layer_name})...")

        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        feature_extractor = create_feature_extractor(
            base_model,
            return_nodes={self.layer_name: 'features'}
        )

        if self.pool_size > 1:
            # Wrap with pooling
            class PooledExtractor(nn.Module):
                def __init__(self, extractor, pool_size):
                    super().__init__()
                    self.extractor = extractor
                    self.pool = nn.MaxPool2d(kernel_size=pool_size)

                def forward(self, x):
                    features = self.extractor(x)['features']
                    return self.pool(features)

            feature_extractor = PooledExtractor(feature_extractor, self.pool_size)

        logger.debug(f"Created ResNet50 feature extractor from layer: {self.layer_name}")
        return feature_extractor

    def _get_transform(self) -> Callable[[Image.Image], torch.Tensor]:
        """Get ImageNet normalization transform."""
        return T.Compose([
            T.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _get_cache_filename(self) -> str:
        """Get cache filename for this configuration."""
        return f"resnet50_{self.layer_name}_pool{self.pool_size}.pkl"

    def _get_model_state(self) -> Dict:
        """Get ResNet50-specific state for saving."""
        return {
            'layer_name': self.layer_name,
            'pool_size': self.pool_size,
        }

    def _get_training_description(self) -> str:
        """Get training pipeline description."""
        return f"ResNet50({self.layer_name}) -> Pool({self.pool_size}) -> PCA({self.n_components}) -> SVM"

    def _should_flatten_features(self) -> bool:
        """ResNet50 intermediate layers output spatial maps that need flattening."""
        return True

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        n_classes = len(self._class_mapping) if self._class_mapping else 0
        return (f"ResNet50Identifier(layer={self.layer_name}, pool={self.pool_size}, "
                f"pca={self.n_components}, classes={n_classes}, {status})")


class MegaDescriptorIdentifier(ElephantIdentifier):
    """Elephant identifier using MegaDescriptor-L-384 feature extraction.

    Uses the MegaDescriptor foundation model for wildlife re-identification,
    then applies PCA + SVM for classification.

    MegaDescriptor-L-384 outputs 1536-dimensional feature vectors directly,
    so no flattening is needed.
    """

    MODEL_TYPE = "megadescriptor"
    INPUT_SIZE = 384

    def __init__(self, n_components: int = 1024):
        """Initialize MegaDescriptor identifier.

        Args:
            n_components: Number of PCA components to retain.
                          MegaDescriptor-L-384 outputs 1536-dim features.
        """
        super().__init__(n_components=n_components)

    def _create_feature_extractor(self) -> nn.Module:
        """Create MegaDescriptor feature extractor."""
        logger.info("Loading MegaDescriptor-L-384 model...")
        model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
        logger.debug("MegaDescriptor loaded successfully")
        return model

    def _get_transform(self) -> Callable[[Image.Image], torch.Tensor]:
        """Get MegaDescriptor normalization transform."""
        return T.Compose([
            T.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def _get_cache_filename(self) -> str:
        """Get cache filename for this configuration."""
        return "megadescriptor_l384.pkl"

    def _get_model_state(self) -> Dict:
        """Get MegaDescriptor-specific state for saving."""
        return {}  # No additional state needed

    def _get_training_description(self) -> str:
        """Get training pipeline description."""
        return f"MegaDescriptor-L-384 -> PCA({self.n_components}) -> SVM"

    def _should_flatten_features(self) -> bool:
        """MegaDescriptor outputs 1D feature vectors, no flattening needed."""
        return False

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        n_classes = len(self._class_mapping) if self._class_mapping else 0
        return (f"MegaDescriptorIdentifier(pca={self.n_components}, "
                f"classes={n_classes}, {status})")
