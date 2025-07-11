import os
import requests
from typing import Literal
from dotenv import load_dotenv
from pprint import pprint
from tqdm import tqdm
import re

load_dotenv()

API_KEY = os.environ["PEXELS_API_KEY"]

def search(query: str, *, image_count: int = 15, size: Literal["tiny", "small", "medium", "large", "large2x"] = "small") -> list[str]:
    """
    Search for images on Pexels

    Args:
        query (str): The query to search for.
        image_count (int): The number of images to return.
        size (Literal["tiny", "small", "medium", "large", "large2x"]): The size of images to return. Options are tiny, small, medium, large, and large2x.

    Returns:
        list[str]: A list of image URLs.
    """
    MAX_PER_PAGE = 80
    
    page = 1
    images: list[str] = []

    while len(images) < image_count:
        url = f"https://api.pexels.com/v1/search?query={query}&page={page}&per_page={MAX_PER_PAGE}"
        headers = {"Authorization": API_KEY}
        try:
            response = requests.get(url, headers=headers).json()
        except Exception as e:
            print(f"Error getting images: {e}")
            break

        images.extend([photo["src"][size] for photo in response["photos"]])
        page += 1
    
    return images[:image_count]

def get_photo_by_id(photo_id: int, *, size: Literal["tiny", "small", "medium", "large", "large2x"] = "small") -> str | None:
    """
    Get a specific photo by its ID from Pexels.

    Args:
        photo_id (int): The ID of the photo to retrieve.
        size (Literal["tiny", "small", "medium", "large", "large2x"]): The size of image to return.

    Returns:
        str | None: The image URL if found, None otherwise.
    """
    url = f"https://api.pexels.com/v1/photos/{photo_id}"
    headers = {"Authorization": API_KEY}
    
    try:
        response = requests.get(url, headers=headers).json()
        return response["src"][size]
    except Exception as e:
        print(f"Error getting photo {photo_id}: {e}")
        return None

def redownload_existing_images_large(output_dir: str = "elephant_head_training") -> None:
    """
    Redownload existing images in the directory in large size.

    Args:
        output_dir (str): The directory containing the existing images.
    """
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist.")
        return

    # Get all image files in the directory
    image_files = [f for f in os.listdir(output_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
    
    if not image_files:
        print(f"No image files found in {output_dir}")
        return

    print(f"Found {len(image_files)} existing images to redownload in large size")

    # Extract photo IDs from filenames (format: pexels-photo-{ID}.jpeg)
    photo_ids = []
    for filename in image_files:
        match = re.search(r'pexels-photo-(\d+)', filename)
        if match:
            photo_ids.append(int(match.group(1)))

    if not photo_ids:
        print("No valid Pexels photo IDs found in filenames")
        return

    print(f"Extracted {len(photo_ids)} photo IDs")

    # Download each photo in large size
    downloaded_count = 0
    for photo_id in tqdm(photo_ids, desc="Redownloading images in large size"):
        try:
            image_url = get_photo_by_id(photo_id, size="large")
            if image_url:
                # Create new filename with large size
                filename = f"pexels-photo-{photo_id}-large.jpeg"
                filepath = os.path.join(output_dir, filename)
                
                # Download the image
                response = requests.get(image_url)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                
                downloaded_count += 1
            else:
                print(f"Could not retrieve photo {photo_id}")
                
        except Exception as e:
            print(f"Error downloading photo {photo_id}: {e}")

    print(f"Successfully downloaded {downloaded_count} images in large size")

def download_images(query: str, output_dir: str = "elephant_head_training", *, image_count: int = 15, size: Literal["tiny", "small", "medium", "large", "large2x"] = "small") -> None:
    """
    Download images from Pexels and save them to a local directory.

    Args:
        query (str): The query to search for.
        output_dir (str): The directory to save the downloaded images to.
        image_count (int): The number of images to download. Defaults to 15
        size (Literal["tiny", "small", "medium", "large", "large2x"]): The size of images to download. Defaults to small.

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = search(query, image_count=image_count, size=size)
    print(len(images))


    downloaded_images = 0
    for image in tqdm(images, desc="Downloading images"):
        try:
            filename = image.split("/")[-1].split("?")[0]
            downloaded_images += 1

            if os.path.exists(os.path.join(output_dir, filename)):
                print(f"Image {filename} already exists")
                continue

            response = requests.get(image)
            with open(os.path.join(output_dir, filename), "wb") as f:
                f.write(response.content)

        except Exception as e:
            print(f"Error downloading image {filename}:\n{e}")
        
    print(f"Downloaded {downloaded_images} images")

def clear_images(output_dir: str = "elephant_head_training") -> None:
    """
    Clear all images from a directory.

    Args:
        output_dir (str): The directory to clear.
    """
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))

def delete_non_large_images(output_dir: str = "elephant_head_training") -> None:
    """
    Delete all images that are not large size (keeping only images with '-large' in filename).

    Args:
        output_dir (str): The directory containing the images.
    """
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist.")
        return

    # Get all image files in the directory
    image_files = [f for f in os.listdir(output_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
    
    if not image_files:
        print(f"No image files found in {output_dir}")
        return

    # Filter out large images (keep only those with '-large' in filename)
    non_large_images = [f for f in image_files if '-large' not in f]
    
    if not non_large_images:
        print("No non-large images found to delete")
        return

    print(f"Found {len(non_large_images)} non-large images to delete")

    # Delete non-large images
    deleted_count = 0
    for filename in tqdm(non_large_images, desc="Deleting non-large images"):
        try:
            filepath = os.path.join(output_dir, filename)
            os.remove(filepath)
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {filename}: {e}")

    print(f"Successfully deleted {deleted_count} non-large images")

# Redownload existing images in large size
#redownload_existing_images_large("elephant_head_training")

# Delete all non-large images
delete_non_large_images("elephant_head_training")

