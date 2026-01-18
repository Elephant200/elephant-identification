"""Preprocessing pipeline for appearance-based elephant identification.

Handles image preprocessing, cropping, and train/test splitting.
"""
import argparse
import json
import logging
import os
import random
from typing import Tuple

import cv2
import pandas as pd
from tqdm import tqdm

from appearance.crop import detect_face, remove_background
from utils import get_all_images

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_SIZE = 224
PADDING = 0.05


def clamp(x: int, lower: int, upper: int) -> int:
    return max(lower, min(x, upper))


def extract_name_from_filepath(filepath: str) -> str:
    """Extract elephant name from filename (part before first underscore)."""
    filename = os.path.basename(filepath)
    return filename.split("_")[0]


def preprocess_images(
    image_paths: list[str],
    output_dir: str = "dataset/appearance_faces",
    force: bool = False
) -> list[dict]:
    """
    Preprocess images by detecting faces, removing background, and cropping.

    Args:
        image_paths: List of paths to raw elephant images.
        output_dir: Directory to save preprocessed face images.
        force: Whether to force reprocessing of existing images.

    Returns:
        List of dicts with 'name', 'filepath', and 'original_path' keys.
    """
    os.makedirs(output_dir, exist_ok=True)
    processed_images: list[dict] = []
    failed_images: list[str] = []

    existing_files = set()
    if not force:
        existing_files = {f for f in os.listdir(output_dir) if f.endswith(".jpg")}

    for image_path in tqdm(image_paths, desc="Preprocessing images"):
        name = extract_name_from_filepath(image_path)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        if not force and f"{base_filename}.jpg" in existing_files:
            processed_images.append({
                "name": name,
                "filepath": os.path.join(output_dir, f"{base_filename}.jpg"),
                "original_path": image_path
            })
            continue

        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            failed_images.append(image_path)
            continue

        try:
            faces = detect_face(image_path)
        except Exception as e:
            logger.warning(f"Face detection failed for {image_path}: {e}")
            failed_images.append(image_path)
            continue

        if not faces:
            logger.warning(f"No face detected in {image_path}")
            failed_images.append(image_path)
            continue

        face = max(faces, key=lambda f: f.get("confidence", 0))

        x_center = face["x"]
        y_center = face["y"]
        width = face["width"]
        height = face["height"]

        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        pad_w = int(width * PADDING)
        pad_h = int(height * PADDING)
        x_min = clamp(x_min - pad_w, 0, image.shape[1])
        y_min = clamp(y_min - pad_h, 0, image.shape[0])
        x_max = clamp(x_max + pad_w, 0, image.shape[1])
        y_max = clamp(y_max + pad_h, 0, image.shape[0])

        cropped = image[y_min:y_max, x_min:x_max]

        try:
            crop_bbox = {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
            cropped = remove_background(cropped, image_path=image_path, crop_bbox=crop_bbox)
        except Exception as e:
            logger.warning(f"Background removal failed for {image_path}: {e}")
            failed_images.append(image_path)
            continue

        resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE))

        out_path = os.path.join(output_dir, f"{base_filename}.jpg")
        cv2.imwrite(out_path, resized)

        processed_images.append({
            "name": name,
            "filepath": out_path,
            "original_path": image_path
        })

    if failed_images:
        logger.info(f"Failed to process {len(failed_images)} images")
        logger.info(failed_images)

    return processed_images


def split_dataset_by_elephant(
    data: pd.DataFrame,
    ratio: float = 0.67,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into training and testing sets on a per-elephant basis."""
    random.seed(random_seed)

    train_data = []
    test_data = []

    for elephant_name in data["name"].unique():
        subset = data[data["name"] == elephant_name]
        filepaths = subset["filepath"].tolist()
        random.shuffle(filepaths)

        split_idx = max(1, round(len(filepaths) * ratio))
        if split_idx >= len(filepaths):
            split_idx = len(filepaths) - 1

        train_files = filepaths[:split_idx]
        test_files = filepaths[split_idx:]

        for fp in train_files:
            row = subset[subset["filepath"] == fp].iloc[0]
            train_data.append(row.to_dict())

        for fp in test_files:
            row = subset[subset["filepath"] == fp].iloc[0]
            test_data.append(row.to_dict())

    return pd.DataFrame(train_data), pd.DataFrame(test_data)


def generate_splits(
    processed_images: list[dict],
    metadata_dir: str,
    min_images: int,
    ratio: float,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate train/test splits and save metadata files."""
    os.makedirs(metadata_dir, exist_ok=True)

    df = pd.DataFrame(processed_images)
    logger.info(f"Total preprocessed images: {len(df)}")

    elephant_counts = df["name"].value_counts()
    valid_elephants = elephant_counts[elephant_counts >= min_images].index
    df = df[df["name"].isin(valid_elephants)].copy()

    logger.info(f"After filtering (>= {min_images} images): {len(df)} images from {df['name'].nunique()} elephants")

    if df.empty:
        raise ValueError(f"No elephants have >= {min_images} images")

    names = list(df["name"].unique())
    random.seed(random_seed)
    random.shuffle(names)
    class_mapping = {name: i for i, name in enumerate(names)}

    class_mapping_path = os.path.join(metadata_dir, "class_mapping.json")
    with open(class_mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=4)
    logger.info(f"Saved class mapping to {class_mapping_path}")

    train_df, test_df = split_dataset_by_elephant(df, ratio=ratio, random_seed=random_seed)

    train_path = os.path.join(metadata_dir, "train.csv")
    test_path = os.path.join(metadata_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Train set: {len(train_df)} images from {train_df['name'].nunique()} elephants")
    logger.info(f"Test set: {len(test_df)} images from {test_df['name'].nunique()} elephants")
    logger.info(f"Saved to {train_path} and {test_path}")

    return train_df, test_df


def preprocess(
    input_dir: str,
    output_dir: str = "dataset/appearance_faces",
    metadata_dir: str = "dataset/appearance_metadata",
    min_images: int = 9,
    ratio: float = 0.67,
    force: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline: detect faces, remove background, crop, split.

    Args:
        input_dir: Directory containing raw elephant images.
        output_dir: Directory to save preprocessed face images.
        metadata_dir: Directory to save train/test CSVs and class mapping.
        min_images: Minimum images per elephant to include.
        ratio: Train/test split ratio.
        force: Force reprocessing of existing images.

    Returns:
        Tuple of (train_df, test_df).
    """
    image_paths = get_all_images(input_dir)
    logger.info(f"Found {len(image_paths)} images in {input_dir}")

    processed_images = preprocess_images(image_paths, output_dir, force)
    logger.info(f"Preprocessed {len(processed_images)} images")

    train_df, test_df = generate_splits(
        processed_images,
        metadata_dir,
        min_images,
        ratio
    )

    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess elephant images for appearance-based identification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing raw elephant images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/appearance_faces",
        help="Directory to save preprocessed face images (default: dataset/appearance_faces)"
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="dataset/appearance_metadata",
        help="Directory to save train/test CSVs (default: dataset/appearance_metadata)"
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=9,
        help="Minimum images per elephant to include in dataset (default: 9)"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.67,
        help="Train/test split ratio (default: 0.67)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of existing images"
    )

    args = parser.parse_args()

    train_df, test_df = preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        min_images=args.min_images,
        ratio=args.ratio,
        force=args.force
    )

    print(f"\nPreprocessing complete!")
    print(f"  Preprocessed faces saved to: {args.output_dir}")
    print(f"  Train set: {len(train_df)} images from {train_df['name'].nunique()} elephants")
    print(f"  Test set: {len(test_df)} images from {test_df['name'].nunique()} elephants")
    print(f"  Metadata saved to: {args.metadata_dir}")
