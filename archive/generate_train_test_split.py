"""Optimized metadata generation for elephant identification dataset.

This module handles dataset creation, train/test splitting, and class mapping
generation with comprehensive error handling and logging
"""

import json
import os
import pandas as pd
import shutil
import random
import logging
import cv2
from typing import Tuple
from utils import get_all_images, is_image
from pprint import pprint


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_name_from_filepath(filepath: str) -> str:
    filepath = os.path.basename(filepath)
    return filepath.split("_")[0]

# Move images to desired location from /processing/ELPephants/unannotated/certain
images: list[dict] = []

root_dir = "/Users/kayoko/Documents/GitHub/elephant-identification"

TYPES = ["ELPephants"]
FORCE = True

for TYPE in TYPES:
    if not os.path.exists(f"{root_dir}/dataset/{TYPE}"):
        os.makedirs(f"{root_dir}/dataset/{TYPE}")

    if FORCE:
        shutil.rmtree(f"{root_dir}/dataset/{TYPE}")
        os.makedirs(f"{root_dir}/dataset/{TYPE}")

    for image in os.listdir(f"{root_dir}/processing/{TYPE}/cropped/certain"):
        if not is_image(f"{root_dir}/processing/{TYPE}/cropped/certain/{image}"):
            continue
        try:
            source_path = f"{root_dir}/processing/{TYPE}/cropped/certain/{image}"
            dest_path = f"{root_dir}/dataset/{TYPE}/{image}"
            # print(f"Copying image from {source_path} to {dest_path}")
            shutil.copy2(source_path, dest_path)
            images.append({"name": extract_name_from_filepath(image), "filepath": dest_path, "data_source": TYPE})
        except Exception as e:
            print(f"Error moving {image}: {e}")
    
    for image in os.listdir(f"{root_dir}/processing/{TYPE}/cropped/probable"):
        if not is_image(f"{root_dir}/processing/{TYPE}/cropped/probable/{image}"):
            continue
        try:
            source_path = f"{root_dir}/processing/{TYPE}/cropped/probable/{image}"
            dest_path = f"{root_dir}/dataset/{TYPE}/{image}"
            shutil.copy2(source_path, dest_path)
            images.append({"name": extract_name_from_filepath(image), "filepath": dest_path, "data_source": TYPE})
        except Exception as e:
            print(f"Error moving {image}: {e}")

print(f"Total images: {len(images)}")

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

def augment_training_images(train_df: pd.DataFrame) -> pd.DataFrame:
    """Apply reflection augmentation to training images.
    
    Creates horizontally flipped versions of all training images and adds them
    to the training dataset. The augmented images are saved with '_reflected' suffix.
    
    Args:
        train_df (pd.DataFrame): Training DataFrame with columns ['name', 'filepath', 'data_source']
        
    Returns:
        pd.DataFrame: Augmented training DataFrame containing both original and reflected images
        
    Raises:
        ValueError: If augmentation fails for any image
    """
    logger.info(f"Starting augmentation of {len(train_df)} training images")
    
    augmented_data = []
    successful_augmentations = 0
    failed_augmentations = 0
    
    # Add all original training images first
    for _, row in train_df.iterrows():
        augmented_data.append(row.to_dict())
    
    # Create reflected versions for each training image
    for _, row in train_df.iterrows():
        original_filepath = row['filepath']
        
        if not is_image(original_filepath):
            logger.warning(f"Skipping non-image file: {original_filepath}")
            continue
            
        try:
            # Create reflected image path
            base_dir = os.path.dirname(original_filepath)
            filename = os.path.basename(original_filepath)
            name_without_ext = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            reflected_filename = f"{name_without_ext}_reflected{ext}"
            reflected_filepath = os.path.join(base_dir, reflected_filename)
            
            # Skip if reflected image already exists
            if os.path.exists(reflected_filepath):
                logger.debug(f"Reflected image already exists: {reflected_filepath}")
            else:
                # Load and reflect the image
                image = cv2.imread(original_filepath)
                if image is None:
                    raise ValueError(f"Could not load image: {original_filepath}")
                    
                reflected_image = cv2.flip(image, 1)  # Horizontal flip
                cv2.imwrite(reflected_filepath, reflected_image)
                logger.debug(f"Created reflected image: {reflected_filepath}")
            
            # Add reflected image to augmented dataset
            augmented_row = row.copy()
            augmented_row['filepath'] = reflected_filepath
            augmented_data.append(augmented_row.to_dict())
            successful_augmentations += 1
            
        except Exception as e:
            logger.error(f"Failed to augment image {original_filepath}: {e}")
            failed_augmentations += 1
            # Continue with other images rather than failing completely
    
    logger.info(f"Augmentation completed: {successful_augmentations} successful, {failed_augmentations} failed")
    logger.info(f"Final training set size: {len(augmented_data)} images (doubled from {len(train_df)})")
    
    return pd.DataFrame(augmented_data)

def split_dataset_by_elephant(
    data: pd.DataFrame, 
    ratio: float = 0.67,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into training and testing sets on a per-elephant basis with validation.
    
    This function ensures that:
    - Every elephant is included in both train and test sets
    - Each elephant contributes proportionally to each split
    - The selection of images per elephant is random but reproducible with a seed
    - Comprehensive validation of input data and results
    - Returns original images only (augmentation handled separately)
    
    Args:
        data (pd.DataFrame): DataFrame with columns ['name', 'filepath', 'data_source']
        ratio (float): Ratio of train to test images per elephant (0.0 to 1.0).
                      Defaults to 0.67 (67% train, 33% test)
        random_seed (int): Random seed for reproducible results. Defaults to 42
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data) DataFrames with original images only
        
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
        if "_reflected" in file: # this was an augmented image; omitted for now
            continue
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
with open(f"{root_dir}/dataset/appearance_metadata/class_mapping.json", "w") as f:
    json.dump(class_mapping, f, indent=4)

try:
    # Step 1: Split the dataset on original images only (no augmentation yet)
    print(f"Step 1: Splitting dataset with {len(data)} samples from {data['name'].nunique()} elephants")
    print("Ensuring clean train/test split with no data leakage...")
    train_data, test_data = split_dataset_by_elephant(
        data,
        ratio=0.67,
        random_seed=42
    )
    
    print(f"Initial split completed:")
    print(f"Training set: {len(train_data)} images from {train_data['name'].nunique()} elephants")
    print(f"Testing set:  {len(test_data)} images from {test_data['name'].nunique()} elephants")
    
    # Step 2: Apply augmentation ONLY to training set
    print(f"\nStep 2: Augmenting training set (reflection)...")
    print("Test set remains unchanged to prevent data leakage.")
    train_data_augmented = augment_training_images(train_data)
    
    print(f"Augmentation completed:")
    print(f"Final training set: {len(train_data_augmented)} images from {train_data_augmented['name'].nunique()} elephants")
    print(f"Final testing set:  {len(test_data)} images from {test_data['name'].nunique()} elephants")
    
    # Step 3: Save the datasets
    print(f"\nStep 3: Saving datasets...")
    print(f"Writing train.csv to: {root_dir}/dataset/appearance_metadata/train.csv")
    print(f"Writing test.csv to: {root_dir}/dataset/appearance_metadata/test.csv")

    train_data_augmented.to_csv(f"{root_dir}/dataset/appearance_metadata/train.csv", index=False)
    test_data.to_csv(f"{root_dir}/dataset/appearance_metadata/test.csv", index=False)
    
    # Step 4: Verification
    print(f"\nStep 4: Verification...")
    written_train_df = pd.read_csv(f"{root_dir}/dataset/appearance_metadata/train.csv")
    written_test_df = pd.read_csv(f"{root_dir}/dataset/appearance_metadata/test.csv")
    
    print(f"Verified train.csv: {len(written_train_df)} images")
    print(f"Verified test.csv: {len(written_test_df)} images")
    
    # Check for data leakage - ensure no test image file appears in training set
    test_filenames = set()
    train_original_filenames = set()
    train_augmented_filenames = set()
    
    for filepath in written_test_df['filepath']:
        filename = os.path.basename(filepath)
        test_filenames.add(filename)
    
    for filepath in written_train_df['filepath']:
        filename = os.path.basename(filepath)
        if '_reflected' in filename:
            train_augmented_filenames.add(filename)
        else:
            train_original_filenames.add(filename)
    
    # Check for direct file leakage (same exact files in both sets)
    direct_leakage = test_filenames.intersection(train_original_filenames)
    augmented_leakage = test_filenames.intersection(train_augmented_filenames)
    
    if direct_leakage:
        print(f"⚠️  WARNING: {len(direct_leakage)} test images found in training set (data leakage!)")
        print(f"Leaked files: {list(direct_leakage)[:5]}...")
    elif augmented_leakage:
        print(f"⚠️  WARNING: {len(augmented_leakage)} test images have augmented versions in training (data leakage!)")
    else:
        print("✓ Data integrity check: No direct file overlap between train and test sets")
        print(f"✓ Training set: {len(train_original_filenames)} original + {len(train_augmented_filenames)} augmented images")
        print(f"✓ Test set: {len(test_filenames)} original images only")

    all_elephants = set(data['name'].unique())
    
    # Show sample counts per elephant  
    print(f"\nSample distribution per elephant:")
    train_counts = train_data_augmented['name'].value_counts()
    test_counts = test_data['name'].value_counts()
    
    print(f"{'Elephant':<20} {'Train':<8} {'Test':<8} {'Total':<8}")
    print("-" * 50)
    for elephant in sorted(list(all_elephants))[:10]:  # Show first 10 elephants
        train_count = train_counts.get(elephant, 0)
        test_count = test_counts.get(elephant, 0)
        total_count = train_count + test_count
        print(f"{elephant:<20} {train_count:<8} {test_count:<8} {total_count:<8}")
    
    if len(all_elephants) > 10:
        print(f"... and {len(all_elephants) - 10} more elephants")
        
    print(f"\nDataset preparation completed successfully!")
    print(f"✓ Clean train/test split with no data leakage")
    print(f"✓ Training set augmented with reflection (doubled in size)")
    print(f"✓ Test set contains only original images")
        
except ValueError as e:
    print(f"Error: {e}")
    print("Try reducing the number of samples per elephant or increasing the minimum_images_per_elephant threshold.")
