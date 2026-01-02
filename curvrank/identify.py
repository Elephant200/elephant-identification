import os
import numpy as np
from tqdm import tqdm
from typing import Literal

from curvrank.contour import get_contour
from curvrank.curve import curvature, curvature_descriptors
from curvrank.lnbnn import (
    aggregate_descriptors_by_scale,
    build_indices_for_view,
    identify_query,
    rank_results,
)
from curvrank.preprocess import preprocess_images
from utils import get_all_images, print_with_padding


DEFAULT_SCALES = np.array([0.02, 0.04, 0.06, 0.08], dtype=np.float32)
DEFAULT_CURV_LENGTH = 1024
DEFAULT_FEAT_DIM = 32
DEFAULT_NUM_KEYPOINTS = 32


def extract_descriptors_from_image(
    image_path: str,
    scales: np.ndarray = DEFAULT_SCALES,
    curv_length: int = DEFAULT_CURV_LENGTH,
    feat_dim: int = DEFAULT_FEAT_DIM,
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
) -> tuple[list[dict[float, np.ndarray]], list[Literal["left", "right"]]]:
    """
    Extract curvature descriptors from an image.

    Args:
        image_path: Path to the preprocessed image.
        scales: Curvature scales as fraction of contour extent.
        curv_length: Length for resampled curvature.
        feat_dim: Dimension of feature descriptors.
        num_keypoints: Number of keypoints to extract.

    Returns:
        List of descriptor dictionaries (one per ear in the image).
        List of views ("left" or "right") for each ear.
    """
    try:
        contours, views = get_contour(image_path)
    except Exception as e:
        print(f"Error extracting contour from {image_path}: {e}")
        return [], []

    descriptors_list = []
    for contour in contours:
        curv = curvature(contour, scales)
        descriptors = curvature_descriptors(
            contour, curv, scales, curv_length, feat_dim, num_keypoints
        )
        descriptors_list.append(descriptors)

    return descriptors_list, views


def build_database(
    image_paths: list[str],
    names: list[str],
    output_dir: str = "curvrank/indices",
    scales: np.ndarray = DEFAULT_SCALES,
) -> tuple[dict[str, dict[float, str]], dict[str, dict[float, tuple[np.ndarray, np.ndarray]]]]:
    """
    Build LNBNN indices from a database of images.

    Args:
        image_paths: Paths to preprocessed database images.
        names: Individual names corresponding to each image.
        output_dir: Directory to save index files.
        scales: Curvature scales.

    Returns:
        Dictionary mapping view -> scale -> index file path.
        Dictionary mapping view -> scale -> (descriptors, names).
    """
    assert len(image_paths) == len(names), f'len(image_paths) == {len(image_paths)} != len(names) == {len(names)}'

    left_descriptors: list[dict[float, np.ndarray]] = []
    left_names: list[str] = []
    right_descriptors: list[dict[float, np.ndarray]] = []
    right_names: list[str] = []

    print("Extracting descriptors from database images...")
    for image_path, name in tqdm(list(zip(image_paths, names)), desc="Processing"):
        desc_list, views = extract_descriptors_from_image(image_path, scales)

        for desc, view in zip(desc_list, views):
            if view == "left":
                left_descriptors.append(desc)
                left_names.append(name)
            else:
                right_descriptors.append(desc)
                right_names.append(name)

    index_paths = {}
    lnbnn_data = {}

    if left_descriptors:
        print(f"\nBuilding left ear indices ({len(left_descriptors)} samples)...")
        left_data = aggregate_descriptors_by_scale(left_descriptors, left_names, scales)
        left_index_paths = build_indices_for_view(left_data, "left", output_dir)
        index_paths["left"] = left_index_paths
        lnbnn_data["left"] = left_data

    if right_descriptors:
        print(f"\nBuilding right ear indices ({len(right_descriptors)} samples)...")
        right_data = aggregate_descriptors_by_scale(right_descriptors, right_names, scales)
        right_index_paths = build_indices_for_view(right_data, "right", output_dir)
        index_paths["right"] = right_index_paths
        lnbnn_data["right"] = right_data

    return index_paths, lnbnn_data


def identify(
    query_image_path: str,
    index_paths: dict[str, dict[float, str]],
    lnbnn_data: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]],
    scales: np.ndarray = DEFAULT_SCALES,
    k: int = 2,
) -> dict[str, dict[str, float]]:
    """
    Identify an individual from a query image.

    Args:
        query_image_path: Path to the query image.
        index_paths: Dictionary mapping view -> scale -> index file path.
        lnbnn_data: Dictionary mapping view -> scale -> (descriptors, names).
        scales: Curvature scales.
        k: Number of neighbors for LNBNN.

    Returns:
        Dictionary mapping view -> {name: score}. More negative = stronger match.
    """
    desc_list, views = extract_descriptors_from_image(query_image_path, scales)

    results = {}
    for desc, view in zip(desc_list, views):
        if view not in index_paths:
            print(f"No index available for {view} ear view")
            continue

        scores = identify_query(desc, index_paths[view], lnbnn_data[view], k)
        results[view] = scores

    return results


def identify_batch(
    query_image_paths: list[str],
    index_paths: dict[str, dict[float, str]],
    lnbnn_data: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]],
    scales: np.ndarray = DEFAULT_SCALES,
    k: int = 2,
) -> list[tuple[str, dict[str, dict[str, float]]]]:
    """
    Identify individuals from multiple query images.

    Args:
        query_image_paths: Paths to query images.
        index_paths: Dictionary mapping view -> scale -> index file path.
        lnbnn_data: Dictionary mapping view -> scale -> (descriptors, names).
        scales: Curvature scales.
        k: Number of neighbors for LNBNN.

    Returns:
        List of (image_path, results) tuples.
    """
    results = []
    for query_path in tqdm(query_image_paths, desc="Identifying"):
        result = identify(query_path, index_paths, lnbnn_data, scales, k)
        results.append((query_path, result))
    return results


def print_identification_results(
    results: dict[str, dict[str, float]], 
    top_n: int = 5
) -> None:
    """
    Print identification results in a readable format.

    Args:
        results: Dictionary mapping view -> {name: score}.
        top_n: Number of top matches to display.
    """
    for view, scores in results.items():
        if not scores:
            print(f"  {view.capitalize()} ear: No matches found")
            continue

        ranked = rank_results(scores)[:top_n]
        print(f"  {view.capitalize()} ear:")
        for i, (name, score) in enumerate(ranked, 1):
            print(f"    {i}. {name}: {score:.4f}")


def pipeline(
    db_image_paths: list[str],
    db_names: list[str],
    query_image_paths: list[str],
    index_dir: str = "curvrank/indices",
) -> list[tuple[str, dict[str, dict[str, float]]]]:
    """
    Full pipeline: build database and identify query images.

    Args:
        db_image_paths: Paths to database images.
        db_names: Individual names for database images.
        query_image_paths: Paths to query images.
        index_dir: Directory to store index files.

    Returns:
        Identification results for each query image.
    """
    print("Building database...")
    index_paths, lnbnn_data = build_database(db_image_paths, db_names, index_dir)

    print("Identifying query images...")
    results = identify_batch(query_image_paths, index_paths, lnbnn_data)

    print("Results:")
    for query_path, result in results:
        print(f"\nQuery: {query_path}")
        print_identification_results(result)

    return results


if __name__ == "__main__":
    image_paths = get_all_images("processing/ELPephants/unannotated/certain")
    preprocessed_paths, views, names = preprocess_images(
        image_paths, output_dir="curvrank/preprocessed"
    )
    print(f"Preprocessed {len(preprocessed_paths)} images")

    db_paths = preprocessed_paths[:-2]
    db_names = [name.split("_")[0] for name in names[:-2]]
    query_paths = preprocessed_paths[-2:]

    results = pipeline(db_paths, db_names, query_paths)
