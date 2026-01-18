"""CurvrankIdentifier model class.

Unified model for curvature-based elephant ear identification using LNBNN.
Maintains separate indices for left and right ears.
"""
import logging
import os
import pickle
from typing import Literal

import numpy as np
from tqdm import tqdm

from .contour import get_contour
from .curve import curvature, curvature_descriptors
from .lnbnn import (
    aggregate_descriptors_by_scale,
    build_indices_for_view,
    identify_query,
    rank_results,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

DEFAULT_SCALES = np.array([0.02, 0.04, 0.06, 0.08], dtype=np.float32)
DEFAULT_CURV_LENGTH = 1024
DEFAULT_FEAT_DIM = 32
DEFAULT_NUM_KEYPOINTS = 32
DEFAULT_INDEX_DIR = 'cache/curvrank/indices'


class CurvrankIdentifier:
    """Curvature-based elephant ear identifier using LNBNN.

    Maintains separate LNBNN indices for left and right ears.
    Same-view matching only: left queries match left database, right matches right.

    Attributes:
        scales: Curvature scales as fraction of contour extent
        curv_length: Length for resampled curvature
        feat_dim: Dimension of feature descriptors
        num_keypoints: Number of keypoints to extract
    """

    def __init__(
        self,
        scales: np.ndarray = DEFAULT_SCALES,
        curv_length: int = DEFAULT_CURV_LENGTH,
        feat_dim: int = DEFAULT_FEAT_DIM,
        num_keypoints: int = DEFAULT_NUM_KEYPOINTS
    ):
        """Initialize the identifier with hyperparameters.

        Args:
            scales: Curvature scales as fraction of contour extent
            curv_length: Length for resampled curvature
            feat_dim: Dimension of feature descriptors
            num_keypoints: Number of keypoints to extract
        """
        self.scales = scales
        self.curv_length = curv_length
        self.feat_dim = feat_dim
        self.num_keypoints = num_keypoints

        self._left_index_paths: dict[float, str] | None = None
        self._right_index_paths: dict[float, str] | None = None
        self._left_lnbnn_data: dict[float, tuple[np.ndarray, np.ndarray]] | None = None
        self._right_lnbnn_data: dict[float, tuple[np.ndarray, np.ndarray]] | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _detect_view(self, image_path: str) -> Literal["left", "right"]:
        """Detect ear view from filename.

        Args:
            image_path: Path to preprocessed ear image

        Returns:
            "left" or "right" based on filename

        Raises:
            ValueError: If view cannot be detected from filename
        """
        filename = os.path.basename(image_path).lower()
        if "_left." in filename or filename.endswith("_left"):
            return "left"
        elif "_right." in filename or filename.endswith("_right"):
            return "right"
        else:
            raise ValueError(f"Cannot detect view from filename: {image_path}. "
                             "Expected *_left.jpg or *_right.jpg")

    def _extract_descriptors(self, image_path: str) -> dict[float, np.ndarray] | None:
        """Extract curvature descriptors from a preprocessed ear image.

        Args:
            image_path: Path to preprocessed ear image

        Returns:
            Dictionary mapping scale to descriptors array, or None if extraction fails
        """
        try:
            contours, _ = get_contour(image_path)
            if not contours:
                logger.warning(f"No contour found in {image_path}")
                return None

            contour = contours[0]
            curv = curvature(contour, self.scales)
            descriptors = curvature_descriptors(
                contour, curv, self.scales,
                self.curv_length, self.feat_dim, self.num_keypoints
            )
            return descriptors
        except Exception as e:
            logger.warning(f"Failed to extract descriptors from {image_path}: {e}")
            return None

    def fit(
        self,
        image_paths: list[str],
        names: list[str],
        index_dir: str = DEFAULT_INDEX_DIR
    ) -> 'CurvrankIdentifier':
        """Build LNBNN indices from preprocessed ear images.

        Args:
            image_paths: Paths to preprocessed ear images (*_left.jpg or *_right.jpg)
            names: Individual names/IDs corresponding to each image
            index_dir: Directory to save index files

        Returns:
            self: The fitted model
        """
        if len(image_paths) != len(names):
            raise ValueError(f"Mismatch: {len(image_paths)} images vs {len(names)} names")

        logger.info(f"Fitting CurvrankIdentifier on {len(image_paths)} images")

        left_descriptors: list[dict[float, np.ndarray]] = []
        left_names: list[str] = []
        right_descriptors: list[dict[float, np.ndarray]] = []
        right_names: list[str] = []

        for image_path, name in tqdm(list(zip(image_paths, names)), desc="Extracting descriptors"):
            try:
                view = self._detect_view(image_path)
            except ValueError as e:
                logger.warning(str(e))
                continue

            desc = self._extract_descriptors(image_path)
            if desc is None:
                continue

            if view == "left":
                left_descriptors.append(desc)
                left_names.append(name)
            else:
                right_descriptors.append(desc)
                right_names.append(name)

        os.makedirs(index_dir, exist_ok=True)

        if left_descriptors:
            logger.info(f"Building left ear indices ({len(left_descriptors)} samples)")
            left_data = aggregate_descriptors_by_scale(left_descriptors, left_names, self.scales)
            self._left_index_paths = build_indices_for_view(left_data, "left", index_dir)
            self._left_lnbnn_data = left_data
        else:
            logger.warning("No left ear samples found")
            self._left_index_paths = {}
            self._left_lnbnn_data = {}

        if right_descriptors:
            logger.info(f"Building right ear indices ({len(right_descriptors)} samples)")
            right_data = aggregate_descriptors_by_scale(right_descriptors, right_names, self.scales)
            self._right_index_paths = build_indices_for_view(right_data, "right", index_dir)
            self._right_lnbnn_data = right_data
        else:
            logger.warning("No right ear samples found")
            self._right_index_paths = {}
            self._right_lnbnn_data = {}

        self._is_fitted = True
        logger.info("Fitting complete")
        return self

    def predict(self, image_path: str, k: int = 2) -> list[tuple[str, float]]:
        """Predict identity for a preprocessed ear image.

        Args:
            image_path: Path to preprocessed ear image
            k: Number of neighbors for LNBNN (paper recommends k=2)

        Returns:
            List of (name, score) tuples sorted by score (most negative = best match)

        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        view = self._detect_view(image_path)
        desc = self._extract_descriptors(image_path)

        if desc is None:
            logger.warning(f"Could not extract descriptors from {image_path}")
            return []

        if view == "left":
            index_paths = self._left_index_paths
            lnbnn_data = self._left_lnbnn_data
        else:
            index_paths = self._right_index_paths
            lnbnn_data = self._right_lnbnn_data

        if not index_paths:
            logger.warning(f"No index available for {view} ear view")
            return []

        scores = identify_query(desc, index_paths, lnbnn_data, k)
        ranked = rank_results(scores)

        return ranked

    def predict_batch(
        self,
        image_paths: list[str],
        k: int = 2
    ) -> list[list[tuple[str, float]]]:
        """Predict identities for multiple preprocessed ear images.

        Args:
            image_paths: Paths to preprocessed ear images
            k: Number of neighbors for LNBNN

        Returns:
            List of prediction lists, one per image
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        results = []
        for image_path in tqdm(image_paths, desc="Predicting"):
            result = self.predict(image_path, k)
            results.append(result)
        return results

    def evaluate(
        self,
        test_paths: list[str],
        test_names: list[str],
        top_k_values: list[int] | None = None,
        k: int = 2
    ) -> dict[str, dict[str, float]]:
        """Evaluate model accuracy on a test set.

        Args:
            test_paths: Paths to preprocessed test ear images
            test_names: True names/IDs for each test image
            top_k_values: List of k values for top-k accuracy
            k: Number of neighbors for LNBNN

        Returns:
            Dict mapping 'top_k' to accuracy metrics
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if top_k_values is None:
            top_k_values = [1, 3, 5, 10]

        if len(test_paths) != len(test_names):
            raise ValueError(f"Mismatch: {len(test_paths)} paths vs {len(test_names)} names")

        predictions = self.predict_batch(test_paths, k)

        accuracies = {f"top_{kv}": {"correct": 0, "total": 0} for kv in top_k_values}

        for true_name, preds in zip(test_names, predictions):
            if not preds:
                continue

            pred_names = [name for name, _ in preds]

            for kv in top_k_values:
                accuracies[f"top_{kv}"]["total"] += 1
                if true_name in pred_names[:kv]:
                    accuracies[f"top_{kv}"]["correct"] += 1

        for key in accuracies:
            total = accuracies[key]["total"]
            correct = accuracies[key]["correct"]
            accuracies[key]["accuracy"] = correct / total if total > 0 else 0.0

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
            'scales': self.scales,
            'curv_length': self.curv_length,
            'feat_dim': self.feat_dim,
            'num_keypoints': self.num_keypoints,
            'left_index_paths': self._left_index_paths,
            'right_index_paths': self._right_index_paths,
            'left_lnbnn_data': self._left_lnbnn_data,
            'right_lnbnn_data': self._right_lnbnn_data,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'CurvrankIdentifier':
        """Load a trained model from disk.

        Args:
            path: Path to the saved model pickle file

        Returns:
            CurvrankIdentifier: Loaded model ready for inference
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        model = cls(
            scales=state['scales'],
            curv_length=state['curv_length'],
            feat_dim=state['feat_dim'],
            num_keypoints=state['num_keypoints']
        )

        model._left_index_paths = state['left_index_paths']
        model._right_index_paths = state['right_index_paths']
        model._left_lnbnn_data = state['left_lnbnn_data']
        model._right_lnbnn_data = state['right_lnbnn_data']
        model._is_fitted = True

        logger.info(f"Model loaded from {path}")
        return model

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        left_count = len(self._left_lnbnn_data.get(self.scales[0], ([], []))[1]) if self._left_lnbnn_data else 0
        right_count = len(self._right_lnbnn_data.get(self.scales[0], ([], []))[1]) if self._right_lnbnn_data else 0
        return (f"CurvrankIdentifier(scales={len(self.scales)}, "
                f"left={left_count}, right={right_count}, {status})")


