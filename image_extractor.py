import sys
from bs4 import BeautifulSoup
import requests
import os
import json
from tqdm import tqdm

base = "https://www.sheldrickwildlifetrust.org/orphans/"

def get_images(orphan_name: str) -> list[str]:
    """
    Given the name of an orphan, return a list of image urls corresponding to the images in the carousel on the orphan's SWT page.

    Args:
        orphan_name (str): The name of the orphan to get images for.

    Returns:
        list[str]: A list of image urls.
    """
    try:
        response = requests.get(base + orphan_name)
        soup = BeautifulSoup(response.text, "html.parser")
        carousel = soup.find("div", {"id": "main-slider"})
        images = carousel.find_all("img", {"class": "aspect__media"})
        return [image["data-flickity-lazyload"] for image in images]
    except Exception as e:
        print(f"Error getting images for {orphan_name}: {e}")
        return []
    
def download_images(orphan_name: str, output_dir: str = "elephant_images") -> int:
    """
    Download the images for an orphan from the SWT website.

    Args:
        orphan_name (str): The name of the orphan to download images for.
        output_dir (str): The directory to save the images to.
    
    Returns:
        int: The number of images downloaded.
    """
    images = get_images(orphan_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(f"{output_dir}/{orphan_name}"):
        os.makedirs(f"{output_dir}/{orphan_name}")

    for i, image in enumerate(images):
        if os.path.exists(f"{output_dir}/{orphan_name}/{orphan_name}_{i}.jpg"):
                continue
        try:
            response = requests.get(image)
            with open(f"{output_dir}/{orphan_name}/{orphan_name}_{i}.jpg", "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"Error downloading image {i} for {orphan_name}: {e}")
    if len(images) > 0:
        #print(f"Successfully downloaded {len(images)} images for {orphan_name}")
        pass
    else:
        print(f"No images found for {orphan_name}")
    return len(images)

def get_orphan_names() -> dict[str, int]:
    """
    Get a dict of all the orphan names with blank image counts on the SWT website.

    Returns:
        dict[str, int]: A dictionary of orphan names and their image counts (initialized to 0).
    """
    orphans = {}
    for i in tqdm(range(1, 18), desc="Getting orphan names"):
        response = requests.get(f"{base}?page={i}")
        soup = BeautifulSoup(response.text, "html.parser")
        orphan_cards = soup.find_all("div", {"class": "nuCard__inner"})
        for card in orphan_cards:
            name = card.find("a")["href"].split("/")[-1]
            orphans[name] = 0
    
    return orphans

try:
    with open("orphans.json", "r") as f:
        orphans = json.load(f)
except FileNotFoundError:
    orphans = get_orphan_names()
    with open("orphans.json", "w") as f:
        json.dump(orphans, f)
print(len(orphans))

print("Press ctrl + c to stop the download at any time.")

try:
    for orphan in tqdm(orphans.keys(), desc="Downloading images"):
        orphans[orphan] = download_images(orphan)
except KeyboardInterrupt:
    print("Execution interrupted by user. Saving progress...")
finally:
    with open("orphans.json", "w") as f:
        json.dump(orphans, f)

print(f"Total images downloaded: {sum(orphans.values())}")
