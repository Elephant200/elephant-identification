"""Generate train/test split for curvrank ear images.

Splits left and right ears separately, ensuring each elephant has a consistent
number of images in train and test sets for each view.
"""

import os
import random
import pandas as pd
from typing import Literal
from utils import get_all_images


def extract_info_from_filename(filepath: str) -> tuple[str, Literal["left", "right"]]:
    """
    Extract elephant ID and view from filename.

    Args:
        filepath: Path to image file (e.g., "1002_Cynthia I front_7Mar2015_left.jpg")

    Returns:
        Tuple of (elephant_id, view)
    """
    filename = os.path.basename(filepath)
    elephant_id = filename.split("_")[0]
    
    if filename.endswith("_left.jpg"):
        view = "left"
    elif filename.endswith("_right.jpg"):
        view = "right"
    else:
        raise ValueError(f"Cannot determine view from filename: {filename}")
    
    return elephant_id, view


def split_by_view(
    data: pd.DataFrame,
    ratio: float = 0.67,
    random_seed: int = 42,
    min_images: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset ensuring each elephant has images in both train and test.

    Args:
        data: DataFrame with columns ['elephant_id', 'filepath', 'view']
        ratio: Fraction of images per elephant for training.
        random_seed: Random seed for reproducibility.
        min_images: Minimum images per elephant to include.

    Returns:
        Tuple of (train_df, test_df)
    """
    random.seed(random_seed)
    
    elephant_counts = data['elephant_id'].value_counts()
    valid_elephants = elephant_counts[elephant_counts >= min_images].index
    data = data[data['elephant_id'].isin(valid_elephants)].copy()
    
    train_rows = []
    test_rows = []
    
    for elephant_id in data['elephant_id'].unique():
        subset = data[data['elephant_id'] == elephant_id]
        filepaths = subset['filepath'].tolist()
        random.shuffle(filepaths)
        
        split_idx = max(1, round(len(filepaths) * ratio))
        if split_idx >= len(filepaths):
            split_idx = len(filepaths) - 1
        
        train_files = filepaths[:split_idx]
        test_files = filepaths[split_idx:]
        
        for fp in train_files:
            row = subset[subset['filepath'] == fp].iloc[0]
            train_rows.append(row.to_dict())
        
        for fp in test_files:
            row = subset[subset['filepath'] == fp].iloc[0]
            test_rows.append(row.to_dict())
    
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def generate_curvrank_split(
    input_dir: str = "dataset/curvrank_ears",
    output_dir: str = "dataset/curvrank_metadata",
    ratio: float = 0.67,
    random_seed: int = 42,
    min_images_per_view: int = 8
) -> None:
    """
    Generate train/test split CSVs for curvrank ear images.

    Creates separate splits for left and right ears, then combines them.
    Ensures each elephant has a consistent number of images in train/test
    for each view they appear in.

    Args:
        input_dir: Directory containing ear images.
        output_dir: Directory to save CSV files.
        ratio: Fraction of images per elephant for training.
        random_seed: Random seed for reproducibility.
        min_images_per_view: Minimum images per elephant per view to include.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = get_all_images(input_dir)
    print(f"Found {len(image_paths)} images in {input_dir}")
    
    data = []
    for filepath in image_paths:
        try:
            elephant_id, view = extract_info_from_filename(filepath)
            data.append({
                "elephant_id": elephant_id,
                "filepath": filepath,
                "view": view
            })
        except ValueError as e:
            print(f"Skipping: {e}")
    
    df = pd.DataFrame(data)
    print(f"Parsed {len(df)} images")
    
    left_df = df[df['view'] == 'left'].copy()
    right_df = df[df['view'] == 'right'].copy()
    
    print(f"\nLeft ears: {len(left_df)} images from {left_df['elephant_id'].nunique()} elephants")
    print(f"Right ears: {len(right_df)} images from {right_df['elephant_id'].nunique()} elephants")
    
    print("\n--- LEFT EAR SPLIT ---")
    left_train, left_test = split_by_view(left_df, ratio, random_seed, min_images_per_view)
    print(f"Left train: {len(left_train)} images from {left_train['elephant_id'].nunique()} elephants")
    print(f"Left test: {len(left_test)} images from {left_test['elephant_id'].nunique()} elephants")
    
    print("\n--- RIGHT EAR SPLIT ---")
    right_train, right_test = split_by_view(right_df, ratio, random_seed, min_images_per_view)
    print(f"Right train: {len(right_train)} images from {right_train['elephant_id'].nunique()} elephants")
    print(f"Right test: {len(right_test)} images from {right_test['elephant_id'].nunique()} elephants")
    
    left_train.to_csv(os.path.join(output_dir, "left_train.csv"), index=False)
    left_test.to_csv(os.path.join(output_dir, "left_test.csv"), index=False)
    right_train.to_csv(os.path.join(output_dir, "right_train.csv"), index=False)
    right_test.to_csv(os.path.join(output_dir, "right_test.csv"), index=False)
    
    train_df = pd.concat([left_train, right_train], ignore_index=True)
    test_df = pd.concat([left_test, right_test], ignore_index=True)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"\n--- COMBINED ---")
    print(f"Train: {len(train_df)} images")
    print(f"Test: {len(test_df)} images")
    
    print(f"\n--- VERIFICATION ---")
    verify_split(left_train, left_test, "left")
    verify_split(right_train, right_test, "right")
    
    print(f"\nFiles saved to {output_dir}:")
    print("  - left_train.csv, left_test.csv")
    print("  - right_train.csv, right_test.csv")
    print("  - train.csv, test.csv (combined)")


def verify_split(train_df: pd.DataFrame, test_df: pd.DataFrame, view: str) -> None:
    """Verify the train/test split has no overlap and consistent elephants."""
    train_files = set(train_df['filepath'])
    test_files = set(test_df['filepath'])
    
    overlap = train_files.intersection(test_files)
    if overlap:
        print(f"  WARNING: {view} has {len(overlap)} files in both train and test!")
    else:
        print(f"  {view}: No file overlap between train and test")
    
    train_elephants = set(train_df['elephant_id'])
    test_elephants = set(test_df['elephant_id'])
    
    if train_elephants == test_elephants:
        print(f"  {view}: All {len(train_elephants)} elephants appear in both train and test")
    else:
        only_train = train_elephants - test_elephants
        only_test = test_elephants - train_elephants
        if only_train:
            print(f"  {view}: {len(only_train)} elephants only in train")
        if only_test:
            print(f"  {view}: {len(only_test)} elephants only in test")


def print_distribution(output_dir: str = "dataset/curvrank_metadata") -> None:
    """Print distribution statistics for the generated split."""
    for view in ["left", "right"]:
        train_df = pd.read_csv(os.path.join(output_dir, f"{view}_train.csv"))
        test_df = pd.read_csv(os.path.join(output_dir, f"{view}_test.csv"))
        
        print(f"\n=== {view.upper()} EAR DISTRIBUTION ===")
        
        train_counts = train_df['elephant_id'].value_counts()
        test_counts = test_df['elephant_id'].value_counts()
        
        all_elephants = sorted(set(train_counts.index) | set(test_counts.index))
        
        print(f"{'Elephant':<12} {'Train':<8} {'Test':<8} {'Ratio':<8}")
        print("-" * 40)
        
        for elephant in all_elephants[:10]:
            train_n = train_counts.get(elephant, 0)
            test_n = test_counts.get(elephant, 0)
            total = train_n + test_n
            ratio = train_n / total if total > 0 else 0
            print(f"{elephant:<12} {train_n:<8} {test_n:<8} {ratio:.2f}")
        
        if len(all_elephants) > 10:
            print(f"... and {len(all_elephants) - 10} more elephants")


if __name__ == "__main__":
    generate_curvrank_split(
        input_dir="dataset/curvrank_ears",
        output_dir="dataset/curvrank_metadata",
        ratio=0.67,
        random_seed=42,
        min_images_per_view=8
    )
    print_distribution()

