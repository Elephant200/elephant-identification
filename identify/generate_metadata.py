"""Optimized metadata generation for elephant identification dataset.

This module handles dataset creation, train/test splitting, and class mapping
generation with comprehensive error handling and logging for Apple Silicon optimization.
"""

import json
import os
import pandas as pd
import random
import logging
from typing import List, Tuple, Dict, Set
from utils import get_all_images

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FORCE = True
TYPES = ["ELPephants"]
root_dir = "/Users/kayoko/Documents/GitHub/elephant-identification"

def extract_name_from_filepath(filepath: str) -> str:
    """Extract elephant name from image filepath.
    
    Args:
        filepath (str): Full path to image file
        
    Returns:
        str: Elephant name extracted from filename (part before first underscore)
        
    Raises:
        ValueError: If filepath is empty or invalid
    """
    if not filepath:
        raise ValueError("Filepath cannot be empty")
    
    filename = os.path.basename(filepath)
    if not filename:
        raise ValueError(f"Invalid filepath: {filepath}")
    
    name_parts = filename.split("_")
    if not name_parts[0]:
        raise ValueError(f"Cannot extract name from filename: {filename}")
    
    return name_parts[0]

def split_dataset_by_elephant(
    data: pd.DataFrame, 
    ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into training and testing sets on a per-elephant basis with validation.
    
    This function ensures that:
    - Every elephant is included in both train and test sets
    - Each elephant contributes proportionally to each split
    - The selection of images per elephant is random but reproducible with a seed
    - Comprehensive validation of input data and results
    
    Args:
        data (pd.DataFrame): DataFrame with columns ['name', 'filepath', 'data_source']
        ratio (float): Ratio of train to test images per elephant (0.0 to 1.0).
                      Defaults to 0.8 (80% train, 20% test)
        random_seed (int): Random seed for reproducible results. Defaults to 42
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data) DataFrames
        
    Raises:
        ValueError: If input data is invalid, ratio is out of bounds, or splitting fails
        KeyError: If required columns are missing from input DataFrame
    """
    # Input validation
    if data.empty:
        raise ValueError("Input DataFrame is empty")
    
    required_columns = ['name', 'filepath', 'data_source']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"DataFrame missing required columns: {missing_columns}")
    
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"Ratio must be between 0.0 and 1.0, got {ratio}")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Analyze elephant distribution
    elephant_counts = data['name'].value_counts()
    total_elephants = len(elephant_counts)
    total_images = len(data)
    
    logger.info(f"Splitting dataset: {total_images} images from {total_elephants} elephants")
    logger.info(f"Train/test ratio: {ratio:.1%}/{1-ratio:.1%}")
    
    # Check for elephants with insufficient images
    min_images_required = 2  # At least 1 for train and 1 for test
    insufficient_elephants = elephant_counts[elephant_counts < min_images_required]
    if not insufficient_elephants.empty:
        logger.warning(f"{len(insufficient_elephants)} elephants have < {min_images_required} images")
        logger.warning(f"These elephants may not be well represented: {insufficient_elephants.index.tolist()[:5]}...")
    
    train_data = []
    test_data = []
    elephants_processed = 0
    
    # Process each elephant separately
    for elephant_name in elephant_counts.index:
        try:
            # Get all images for this elephant
            elephant_subset = data[data['name'] == elephant_name]
            elephant_images = elephant_subset['filepath'].tolist()
            
            # Randomly shuffle the images for this elephant
            random.shuffle(elephant_images)
            
            # Calculate split point
            images_per_elephant = len(elephant_images)
            split_point = max(1, round(images_per_elephant * ratio))
            
            train_images = elephant_images[:split_point]
            test_images = elephant_images[split_point:]
            
            # Ensure at least one image in test set if possible
            if not test_images and images_per_elephant > 1:
                test_images = [train_images.pop()]
            
            # Add to respective datasets
            for img_path in train_images:
                elephant_data = data[data['filepath'] == img_path].iloc[0]
                train_data.append(elephant_data.to_dict())
                
            for img_path in test_images:
                elephant_data = data[data['filepath'] == img_path].iloc[0]
                test_data.append(elephant_data.to_dict())
            
            elephants_processed += 1
            
        except Exception as e:
            logger.error(f"Error processing elephant '{elephant_name}': {e}")
            raise ValueError(f"Failed to process elephant '{elephant_name}': {e}")
    
    # Convert back to DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # Validation of results
    if train_df.empty or test_df.empty:
        raise ValueError("Split resulted in empty train or test set")
    
    train_elephants = set(train_df['name'].unique())
    test_elephants = set(test_df['name'].unique())
    
    # Log results
    logger.info(f"Split completed: {len(train_df)} train, {len(test_df)} test images")
    logger.info(f"Elephants in train: {len(train_elephants)}, in test: {len(test_elephants)}")
    
    missing_from_train = test_elephants - train_elephants
    missing_from_test = train_elephants - test_elephants
    
    if missing_from_train:
        logger.warning(f"{len(missing_from_train)} elephants missing from train set")
    if missing_from_test:
        logger.warning(f"{len(missing_from_test)} elephants missing from test set")
    
    return train_df, test_df

data = []

for TYPE in TYPES:
    files = get_all_images(f"{root_dir}/dataset/{TYPE}")
    
    for file in files:
        data.append({"name": extract_name_from_filepath(file), "filepath": file, "data_source": TYPE})
    
data = pd.DataFrame(data)

if FORCE:
    if os.path.exists(f"{root_dir}/dataset/data.csv"):
        os.remove(f"{root_dir}/dataset/data.csv")

# Quick filter comparison
elephant_counts = data['name'].value_counts()
for threshold in [1, 5, 6, 7, 8, 9, 10, 11, 12, 15]:
    remaining = len(elephant_counts[elephant_counts >= threshold])
    images = elephant_counts[elephant_counts >= threshold].sum()
    print(f"≥{threshold:2d} images: {remaining:3d} elephants, {images:4d} images")

minimum_images_per_elephant = 9  # Change this to apply different filter

valid_elephants = elephant_counts[elephant_counts >= minimum_images_per_elephant].index

# Filter the data
print(f"\nApplying filter: ≥{minimum_images_per_elephant} images per elephant")
print(f"Before: {len(data)} images, {data['name'].nunique()} elephants")
data = data[data['name'].isin(valid_elephants)]
print(f"After:  {len(data)} images, {data['name'].nunique()} elephants")

names = data['name'].unique()
random.seed(42)
random.shuffle(names)
class_mapping = {name: i for i, name in enumerate(names)}
with open(f"{root_dir}/dataset/class_mapping.json", "w") as f:
    json.dump(class_mapping, f)

try:
    # Split the dataset: 80% train, 20% test
    print(f"About to split dataset with {len(data)} samples from {data['name'].nunique()} elephants")
    train_data, test_data = split_dataset_by_elephant(
        data,
        ratio=0.8,
        random_seed=42
    )

    print(f"Writing train.csv to: {root_dir}/dataset/train.csv")
    print(f"Train data first few names: {train_data['name'].head().tolist()}")
    print(f"Train data unique names: {sorted(train_data['name'].unique())[:5]}")
    train_data.to_csv(f"{root_dir}/dataset/train.csv", index=False)
    test_data.to_csv(f"{root_dir}/dataset/test.csv", index=False)

    # Verify the file was written correctly
    print("Verifying written file...")
    written_df = pd.read_csv(f"{root_dir}/dataset/train.csv")
    print(f"Written file first few names: {written_df['name'].head().tolist()}")
    print(f"Written file unique names: {sorted(written_df['name'].unique())[:5]}")

    print(f"Split completed successfully!")
    print(f"Training set: {len(train_data)} images from {train_data['name'].nunique()} elephants")
    print(f"Testing set:  {len(test_data)} images from {test_data['name'].nunique()} elephants")

    all_elephants = set(data['name'].unique())
    
    
    # Show sample counts per elephant
    print(f"\nSample counts per elephant:")
    train_counts = train_data['name'].value_counts()
    test_counts = test_data['name'].value_counts()
    
    # Display first 5 elephants as example
    for elephant in list(all_elephants):
        train_count = train_counts.get(elephant, 0)
        test_count = test_counts.get(elephant, 0)
        print(f"  {elephant}: {train_count} train, {test_count} test")
    
    # if len(all_elephants) > 5:
    #     print(f"  ... and {len(all_elephants) - 5} more elephants")
        
except ValueError as e:
    print(f"Error: {e}")
    print("Try reducing the number of samples per elephant or increasing the minimum_images_per_elephant threshold.")
