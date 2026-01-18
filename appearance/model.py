"""ElephantIdentifier model class.

Unified model that wraps ResNet50 feature extraction, PCA dimensionality
reduction, and SVM classification into a single saveable/loadable object.
"""
import logging
import os
import pickle
import time
from typing import Dict, List, Tuple

import keras
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .core import (
    create_feature_extractor,
    extract_features_batch,
    extract_single_image_features,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

DEFAULT_CACHE_DIR = 'cache/appearance/features'


class ElephantIdentifier:
    """Unified elephant identification model.

    Combines ResNet50 feature extraction, StandardScaler, PCA dimensionality
    reduction, and SVM classification into a single model that can be saved
    and loaded for inference.

    Attributes:
        layer_name: ResNet50 layer used for feature extraction
        pool_size: Max pooling size after feature extraction
        n_components: Number of PCA components
        class_mapping: Dict mapping elephant name/ID to class index
        reverse_mapping: Dict mapping class index to elephant name/ID
    """

    def __init__(
        self,
        layer_name: str = 'conv3_block4_2_relu',
        pool_size: int = 6,
        n_components: int = 10000
    ):
        """Initialize the identifier with model hyperparameters.

        Args:
            layer_name: ResNet50 layer name for feature extraction
            pool_size: Max pooling size. Use 1 to disable pooling.
            n_components: Number of PCA components to retain
        """
        self.layer_name = layer_name
        self.pool_size = pool_size
        self.n_components = n_components

        self._feature_extractor: keras.Model | None = None
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

    def _ensure_feature_extractor(self) -> keras.Model:
        """Lazily create the feature extractor."""
        if self._feature_extractor is None:
            self._feature_extractor = create_feature_extractor(
                self.layer_name,
                self.pool_size
            )
        return self._feature_extractor

    def _get_cache_path(self, cache_dir: str, prefix: str = 'train') -> str:
        """Get cache file path for features."""
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(
            cache_dir,
            f"{prefix}_features_{self.layer_name}_pool{self.pool_size}.pkl"
        )

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
        logger.info(f"Training pipeline: ResNet50({self.layer_name}) -> "
                    f"Pool({self.pool_size}) -> PCA({self.n_components}) -> SVM")

        self._class_mapping = class_mapping
        self._reverse_mapping = {v: k for k, v in class_mapping.items()}

        feature_extractor = self._ensure_feature_extractor()
        cache_path = self._get_cache_path(cache_dir, 'train')

        start_time = time.perf_counter()
        raw_features = extract_features_batch(
            train_df,
            feature_extractor,
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

        feature_extractor = self._ensure_feature_extractor()
        raw_features = extract_single_image_features(image_path, feature_extractor)

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
        feature_extractor = self._ensure_feature_extractor()
        cache_path = self._get_cache_path(cache_dir, 'predict')

        raw_features = extract_features_batch(
            df,
            feature_extractor,
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

        feature_extractor = self._ensure_feature_extractor()
        cache_path = self._get_cache_path(cache_dir, 'test')

        start_time = time.perf_counter()
        raw_features = extract_features_batch(
            test_df,
            feature_extractor,
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
            'layer_name': self.layer_name,
            'pool_size': self.pool_size,
            'n_components': self.n_components,
            'scaler': self._scaler,
            'pca': self._pca,
            'svm': self._svm,
            'class_mapping': self._class_mapping,
            'reverse_mapping': self._reverse_mapping,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'ElephantIdentifier':
        """Load a trained model from disk.

        Args:
            path: Path to the saved model pickle file

        Returns:
            ElephantIdentifier: Loaded model ready for inference
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        model = cls(
            layer_name=state['layer_name'],
            pool_size=state['pool_size'],
            n_components=state['n_components']
        )

        model._scaler = state['scaler']
        model._pca = state['pca']
        model._svm = state['svm']
        model._class_mapping = state['class_mapping']
        model._reverse_mapping = state['reverse_mapping']
        model._is_fitted = True

        logger.info(f"Model loaded from {path}")
        return model

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        n_classes = len(self._class_mapping) if self._class_mapping else 0
        return (f"ElephantIdentifier(layer={self.layer_name}, pool={self.pool_size}, "
                f"pca={self.n_components}, classes={n_classes}, {status})")

