import os
import shutil
import random
from tqdm import tqdm

def move_images_batch(source_dir: str = "ELPephants", 
                     dest_dir: str = "next_batch", 
                     count: int = 250,
                     exclude_dir: str = None) -> None:
    """
    Move a specified number of images from source directory to destination directory.
    
    Args:
        source_dir (str): Source directory containing images to move.
        dest_dir (str): Destination directory to move images to.
        count (int): Number of images to move.
        exclude_dir (str): Directory containing images to exclude from selection.
    """
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created destination directory '{dest_dir}'")
    
    # Get all image files from source directory
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(source_dir) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No image files found in '{source_dir}'")
        return
    
    print(f"Found {len(image_files)} images in '{source_dir}'")
    
    # If exclude directory is specified, get list of images to exclude
    exclude_images = set()
    if exclude_dir and os.path.exists(exclude_dir):
        exclude_files = [f for f in os.listdir(exclude_dir) 
                        if f.lower().endswith(image_extensions)]
        exclude_images = set(exclude_files)
        print(f"Excluding {len(exclude_images)} images that are already in '{exclude_dir}'")
    
    # Filter out images that are in the exclude directory
    available_images = [f for f in image_files if f not in exclude_images]
    
    if not available_images:
        print(f"No images available after excluding those in '{exclude_dir}'")
        return
    
    print(f"Available images for selection: {len(available_images)}")
    
    # Determine how many images to move
    images_to_move = min(count, len(available_images))
    
    if images_to_move < count:
        print(f"Warning: Only {len(available_images)} images available, moving all of them")
    
    # Randomly select images to move
    selected_images = random.sample(available_images, images_to_move)
    
    print(f"Moving {images_to_move} images from '{source_dir}' to '{dest_dir}'")
    
    # Move the selected images
    moved_count = 0
    for filename in tqdm(selected_images, desc="Moving images"):
        try:
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            
            # Move the file
            shutil.move(source_path, dest_path)
            moved_count += 1
            
        except Exception as e:
            print(f"Error moving {filename}: {e}")
    
    print(f"Successfully moved {moved_count} images to '{dest_dir}'")
    print(f"Remaining images in '{source_dir}': {len(image_files) - moved_count}")

if __name__ == "__main__":
    # Move 250 images from ELPephants to batch_2, excluding those in batch_1
    move_images_batch("ELPephants", "batch_2", 250, exclude_dir="batch_1")
