"""Preprocessing pipeline for curvrank-based elephant ear identification.

Handles ear detection, cropping, and train/test splitting.
"""
import argparse
import logging
import os
import random
from typing import Literal, Tuple

import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from curvrank.contour import get_contour
from utils import get_all_images

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_SIZE = 432
PADDING = 0.10


def clamp(x: int, lower: int, upper: int) -> int:
    return max(lower, min(x, upper))


def extract_info_from_filename(filepath: str) -> Tuple[str, Literal["left", "right"]]:
    """Extract elephant ID and view from preprocessed ear filename."""
    filename = os.path.basename(filepath)
    elephant_id = filename.split("_")[0]

    if filename.endswith("_left.jpg"):
        view = "left"
    elif filename.endswith("_right.jpg"):
        view = "right"
    else:
        raise ValueError(f"Cannot determine view from filename: {filename}")

    return elephant_id, view


def preprocess_images(
    image_paths: list[str],
    output_dir: str = "dataset/curvrank_ears",
    force: bool = False
) -> Tuple[list[str], list[Literal["left", "right"]], list[str]]:
    """
    Preprocess images by detecting ear contours and cropping.

    For every image, split it into different images, one for each ear.
    Each image is resized to 432x432 for rf-detr inference.

    Args:
        image_paths: List of paths to raw images. Extracts names from the paths.
        output_dir: Directory to save the preprocessed images.
        force: Whether to force reprocessing of images that already exist.

    Returns:
        Tuple of (out_paths, out_views, out_names).
    """
    failed_images: list[str] = []
    os.makedirs(output_dir, exist_ok=True)

    out_paths: list[str] = []
    out_views: list[Literal["left", "right"]] = []
    out_names: list[str] = []

    if not force:
        existing_images = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
        image_paths_to_process = []
        for ip in image_paths:
            base = ip.split("/")[-1].split(".")[0]
            if f"{base}_right.jpg" not in existing_images and f"{base}_left.jpg" not in existing_images:
                image_paths_to_process.append(ip)
            else:
                for ei in existing_images:
                    if ei.startswith(base + "_"):
                        out_paths.append(os.path.join(output_dir, ei))
                        out_views.append(ei.split(".")[0].split("_")[-1])
                        out_names.append("_".join(ei.split(".")[0].split("_")[:-1]))

        if not image_paths_to_process:
            logger.info("No new images to preprocess")
        else:
            logger.info(f"Skipping {len(existing_images)} existing images, processing {len(image_paths_to_process)} new")
        image_paths = image_paths_to_process

    for image_path in tqdm(image_paths, desc="Preprocessing images"):
        name = image_path.split("/")[-1].split("_")[0]
        # assert name.isnumeric(), f"Could not extract name from filepath: {image_path}"
        
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            failed_images.append(image_path)
            continue

        try:
            contours, views = get_contour(image_path)
        except Exception as e:
            logger.warning(f"Error getting contour for {image_path}: {e}")
            failed_images.append(image_path)
            continue

        for contour, view in zip(contours, views):
            try:
                x_min = int(contour[:, 0].min())
                y_min = int(contour[:, 1].min())
                x_max = int(contour[:, 0].max())
                y_max = int(contour[:, 1].max())
                w = x_max - x_min
                h = y_max - y_min

                x_min = clamp(int(x_min - PADDING * w), 0, image.shape[1])
                y_min = clamp(int(y_min - PADDING * h), 0, image.shape[0])
                x_max = clamp(int(x_max + PADDING * w), 0, image.shape[1])
                y_max = clamp(int(y_max + PADDING * h), 0, image.shape[0])

                cropped = image[y_min:y_max, x_min:x_max]
                resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE))

                out_name = f"{image_path.split('/')[-1].split('.')[0]}_{view}"
                out_path = os.path.join(output_dir, f"{out_name}.jpg")
                cv2.imwrite(out_path, resized)

                out_paths.append(out_path)
                out_views.append(view)
                out_names.append(out_name)
            except Exception as e:
                logger.warning(f"Error preprocessing {image_path}: {e}")
                failed_images.append(image_path)
                continue

    if failed_images:
        logger.info(f"Failed to process {len(failed_images)} images")

    return out_paths, out_views, out_names


def split_by_view(
    data: pd.DataFrame,
    ratio: float = 0.67,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset ensuring each elephant has images in both train and test."""
    random.seed(random_seed)

    train_rows = []
    test_rows = []

    for elephant_id in data["elephant_id"].unique():
        subset = data[data["elephant_id"] == elephant_id]
        filepaths = subset["filepath"].tolist()
        random.shuffle(filepaths)

        split_idx = max(1, round(len(filepaths) * ratio))
        if split_idx >= len(filepaths):
            split_idx = len(filepaths) - 1

        train_files = filepaths[:split_idx]
        test_files = filepaths[split_idx:]

        for fp in train_files:
            row = subset[subset["filepath"] == fp].iloc[0]
            train_rows.append(row.to_dict())

        for fp in test_files:
            row = subset[subset["filepath"] == fp].iloc[0]
            test_rows.append(row.to_dict())

    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def filter_by_per_ear_minimum(
    df: pd.DataFrame,
    min_images_per_ear: int
) -> pd.DataFrame:
    """
    Filter to elephants with >= min_images_per_ear for BOTH left AND right ears.

    Args:
        df: DataFrame with columns ['elephant_id', 'filepath', 'view']
        min_images_per_ear: Minimum images per elephant per view.

    Returns:
        Filtered DataFrame.
    """
    left_counts = df[df["view"] == "left"]["elephant_id"].value_counts()
    right_counts = df[df["view"] == "right"]["elephant_id"].value_counts()

    valid_left = set(left_counts[left_counts >= min_images_per_ear].index)
    valid_right = set(right_counts[right_counts >= min_images_per_ear].index)

    valid_elephants = valid_left & valid_right

    logger.info(f"Elephants with >= {min_images_per_ear} left ears: {len(valid_left)}")
    logger.info(f"Elephants with >= {min_images_per_ear} right ears: {len(valid_right)}")
    logger.info(f"Elephants meeting BOTH criteria: {len(valid_elephants)}")

    return df[df["elephant_id"].isin(valid_elephants)].copy()


def generate_splits(
    out_paths: list[str],
    out_views: list[Literal["left", "right"]],
    metadata_dir: str,
    min_images_per_ear: int,
    ratio: float,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate train/test splits for curvrank ear images.

    Creates separate splits for left and right ears, then combines them.

    Args:
        out_paths: List of preprocessed image paths.
        out_views: List of views corresponding to each path.
        metadata_dir: Directory to save CSV files.
        min_images_per_ear: Minimum images per elephant per view.
        ratio: Train/test split ratio.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (left_train, left_test, right_train, right_test).
    """
    os.makedirs(metadata_dir, exist_ok=True)

    data = []
    for filepath, view in zip(out_paths, out_views):
        try:
            elephant_id, _ = extract_info_from_filename(filepath)
            data.append({
                "elephant_id": elephant_id,
                "filepath": filepath,
                "view": view
            })
        except ValueError as e:
            logger.warning(f"Skipping: {e}")

    df = pd.DataFrame(data)
    logger.info(f"Total preprocessed ear images: {len(df)}")

    df = filter_by_per_ear_minimum(df, min_images_per_ear)
    logger.info(f"After filtering: {len(df)} images from {df['elephant_id'].nunique()} elephants")

    if df.empty:
        raise ValueError(f"No elephants have >= {min_images_per_ear} images for both ears")

    left_df = df[df["view"] == "left"].copy()
    right_df = df[df["view"] == "right"].copy()

    logger.info(f"Left ears: {len(left_df)} images from {left_df['elephant_id'].nunique()} elephants")
    logger.info(f"Right ears: {len(right_df)} images from {right_df['elephant_id'].nunique()} elephants")

    left_train, left_test = split_by_view(left_df, ratio, random_seed)
    right_train, right_test = split_by_view(right_df, ratio, random_seed)

    left_train.to_csv(os.path.join(metadata_dir, "left_train.csv"), index=False)
    left_test.to_csv(os.path.join(metadata_dir, "left_test.csv"), index=False)
    right_train.to_csv(os.path.join(metadata_dir, "right_train.csv"), index=False)
    right_test.to_csv(os.path.join(metadata_dir, "right_test.csv"), index=False)

    train_df = pd.concat([left_train, right_train], ignore_index=True)
    test_df = pd.concat([left_test, right_test], ignore_index=True)

    train_df.to_csv(os.path.join(metadata_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(metadata_dir, "test.csv"), index=False)

    logger.info(f"Left train: {len(left_train)}, Left test: {len(left_test)}")
    logger.info(f"Right train: {len(right_train)}, Right test: {len(right_test)}")
    logger.info(f"Combined train: {len(train_df)}, Combined test: {len(test_df)}")
    logger.info(f"Saved to {metadata_dir}")

    return left_train, left_test, right_train, right_test


def preprocess(
    input_dir: str = None,
    input_csv: str = None,
    output_dir: str = "dataset/curvrank_ears",
    metadata_dir: str = "dataset/curvrank_metadata",
    min_images_per_ear: int = 8,
    ratio: float = 0.67,
    force: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline: detect ears, crop, split.

    Args:
        input_dir: Directory containing raw elephant images.
        input_csv: CSV file with filepath column (alternative to input_dir).
        output_dir: Directory to save preprocessed ear images.
        metadata_dir: Directory to save train/test CSVs.
        min_images_per_ear: Minimum images per elephant PER VIEW (must meet for both).
        ratio: Train/test split ratio.
        force: Force reprocessing of existing images.

    Returns:
        Tuple of (train_df, test_df).
    """
    if input_dir:
        image_paths = get_all_images(input_dir)
        logger.info(f"Found {len(image_paths)} images in {input_dir}")
    elif input_csv:
        df = pd.read_csv(input_csv)
        image_paths = df["filepath"].tolist()
        logger.info(f"Found {len(image_paths)} images in {input_csv}")
    else:
        raise ValueError("Must specify either input_dir or input_csv")

    out_paths, out_views, out_names = preprocess_images(image_paths, output_dir, force)
    logger.info(f"Preprocessed {len(out_paths)} ear images")

    left_train, left_test, right_train, right_test = generate_splits(
        out_paths,
        out_views,
        metadata_dir,
        min_images_per_ear,
        ratio
    )

    train_df = pd.concat([left_train, right_train], ignore_index=True)
    test_df = pd.concat([left_test, right_test], ignore_index=True)

    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess elephant images for curvrank identification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing raw elephant images"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        help="CSV file with filepath column containing image paths"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/curvrank_ears",
        help="Directory to save preprocessed ear images (default: dataset/curvrank_ears)"
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="dataset/curvrank_metadata",
        help="Directory to save train/test CSVs (default: dataset/curvrank_metadata)"
    )
    parser.add_argument(
        "--min-images-per-ear",
        type=int,
        default=8,
        help="Minimum images per elephant per ear (must meet for BOTH left AND right, default: 8)"
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

    if not args.input_dir and not args.input_csv:
        parser.error("Must specify either --input-dir or --input-csv")

    train_df, test_df = preprocess(
        input_dir=args.input_dir,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        min_images_per_ear=args.min_images_per_ear,
        ratio=args.ratio,
        force=args.force
    )

    print(f"\nPreprocessing complete!")
    print(f"  Preprocessed ears saved to: {args.output_dir}")
    print(f"  Train set: {len(train_df)} images")
    print(f"  Test set: {len(test_df)} images")
    print(f"  Metadata saved to: {args.metadata_dir}")
