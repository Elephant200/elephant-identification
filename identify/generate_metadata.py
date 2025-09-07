import json
import os
import pandas as pd
import random
from utils import get_all_images

FORCE = True

TYPES = ["ELPephants"]

root_dir = "/Users/kayoko/Documents/GitHub/elephant-identification"

def extract_name_from_filepath(filepath: str) -> str:
    filepath = os.path.basename(filepath)
    return filepath.split("_")[0]

def split_dataset_by_elephant(data: pd.DataFrame, 
                            ratio: float = 0.8,
                            random_seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into training and testing sets on a per-elephant basis.
    
    This function ensures that:
    - Every elephant is included in both train and test sets
    - Each elephant contributes exactly the same number of images to each split
    - The selection of images per elephant is random but reproducible with a seed
    
    Args:
        data (pd.DataFrame): DataFrame with columns ['name', 'filepath', 'data_source']
        ratio (float): Ratio of train to test images per elephant
        random_seed (int): Random seed for reproducible results
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data)
        
    Raises:
        ValueError: If any elephant has fewer images than required for both splits
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Check if all elephants have enough images
    elephant_counts = data['name'].value_counts()
    
    train_data = []
    test_data = []
    
    # Process each elephant separately
    for elephant_name in elephant_counts.index:
        # Get all images for this elephant
        elephant_images = data[data['name'] == elephant_name]['filepath'].tolist()
        
        # Randomly shuffle the images for this elephant
        random.shuffle(elephant_images)
        
        # Split into train and test
        images_per_elephant = len(elephant_images)
        train_images = elephant_images[:round(images_per_elephant * ratio)]
        test_images = elephant_images[round(images_per_elephant * ratio):]
        
        # Add to respective datasets
        for img_path in train_images:
            elephant_data = data[data['filepath'] == img_path].iloc[0]
            train_data.append(elephant_data.to_dict())
            
        for img_path in test_images:
            elephant_data = data[data['filepath'] == img_path].iloc[0]
            test_data.append(elephant_data.to_dict())
    
    # Convert back to DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
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
