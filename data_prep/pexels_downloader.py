import argparse
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import requests
from dotenv import load_dotenv
from tqdm import tqdm

PexelsImageSize = Literal["tiny", "small", "medium", "large", "large2x"]
MAX_PER_PAGE = 80


@dataclass(frozen=True)
class PexelsPhoto:
    id: int
    url: str


def _get_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    env_key = os.getenv("PEXELS_API_KEY")
    if env_key:
        return env_key
    raise RuntimeError(
        "Missing Pexels API key. Set environment variable PEXELS_API_KEY (or pass --api-key)."
    )


def _make_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"Authorization": api_key})
    return session


def _request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
    max_retries: int = 8,
) -> dict[str, Any]:
    backoff_s = 1.0
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=timeout_s)
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 2.0, 30.0)
            continue

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else backoff_s
            time.sleep(min(max(sleep_s, 1.0), 60.0))
            backoff_s = min(backoff_s * 2.0, 30.0)
            continue

        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected response shape: {type(data)}")
        return data

    raise RuntimeError("Failed to fetch JSON from Pexels after retries.")


def iter_search_photos(
    query: str,
    *,
    image_count: int,
    size: PexelsImageSize,
    session: requests.Session,
    per_page: int = MAX_PER_PAGE,
) -> Iterable[PexelsPhoto]:
    """
    Yield Pexels photos for a query, paginating until image_count photos are produced.
    """
    produced = 0
    page = 1
    while produced < image_count:
        data = _request_json(
            session,
            "https://api.pexels.com/v1/search",
            params={"query": query, "page": page, "per_page": per_page},
        )
        photos = data.get("photos", [])
        if not isinstance(photos, list) or not photos:
            return

        for photo in photos:
            if produced >= image_count:
                return
            if not isinstance(photo, dict):
                continue
            photo_id = photo.get("id")
            src = photo.get("src")
            if not isinstance(photo_id, int) or not isinstance(src, dict):
                continue
            url = src.get(size)
            if not isinstance(url, str) or not url:
                continue
            produced += 1
            yield PexelsPhoto(id=photo_id, url=url)

        page += 1


def search(
    query: str, *, image_count: int = 15, size: PexelsImageSize = "small", api_key: str | None = None
) -> list[str]:
    """
    Search for images on Pexels

    Args:
        query (str): The query to search for.
        image_count (int): The number of images to return.
        size (Literal["tiny", "small", "medium", "large", "large2x"]): The size of images to return.
        api_key (str | None): Optional Pexels API key override.

    Returns:
        list[str]: A list of image URLs.
    """
    session = _make_session(_get_api_key(api_key))
    return [p.url for p in iter_search_photos(query, image_count=image_count, size=size, session=session)]


def get_photo_by_id(
    photo_id: int, *, size: PexelsImageSize = "small", api_key: str | None = None
) -> str | None:
    """
    Get a specific photo by its ID from Pexels.

    Args:
        photo_id (int): The ID of the photo to retrieve.
        size (Literal["tiny", "small", "medium", "large", "large2x"]): The size of image to return.

    Returns:
        str | None: The image URL if found, None otherwise.
    """
    try:
        session = _make_session(_get_api_key(api_key))
        response = _request_json(session, f"https://api.pexels.com/v1/photos/{photo_id}")
        src = response.get("src", {})
        if isinstance(src, dict):
            url = src.get(size)
            if isinstance(url, str):
                return url
        return None
    except Exception as e:
        print(f"Error getting photo {photo_id}: {e}")
        return None


def redownload_existing_images_large(
    output_dir: str = "elephant_head_training", *, api_key: str | None = None
) -> None:
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
    session = _make_session(_get_api_key(api_key))
    downloaded_count = 0
    for photo_id in tqdm(photo_ids, desc="Redownloading images in large size"):
        try:
            image_url = get_photo_by_id(photo_id, size="large", api_key=_get_api_key(api_key))
            if image_url:
                # Create new filename with large size
                filename = f"pexels-photo-{photo_id}-large.jpeg"
                filepath = os.path.join(output_dir, filename)
                
                # Download the image
                response = session.get(image_url, timeout=60)
                response.raise_for_status()
                with open(filepath, "wb") as f:
                    f.write(response.content)
                
                downloaded_count += 1
            else:
                print(f"Could not retrieve photo {photo_id}")
                
        except Exception as e:
            print(f"Error downloading photo {photo_id}: {e}")

    print(f"Successfully downloaded {downloaded_count} images in large size")


def _safe_filename(photo_id: int, size: str) -> str:
    return f"pexels-photo-{photo_id}-{size}.jpg"


def _download_streaming(session: requests.Session, url: str, dest_path: str) -> None:
    tmp_path = dest_path + ".part"
    resp = session.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    os.replace(tmp_path, dest_path)


def download_images(
    query: str,
    output_dir: str = "elephant_head_training",
    *,
    image_count: int = 15,
    size: PexelsImageSize = "small",
    api_key: str | None = None,
) -> None:
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
    os.makedirs(output_dir, exist_ok=True)
    session = _make_session(_get_api_key(api_key))

    downloaded = 0
    skipped = 0
    failed = 0

    photos = iter_search_photos(query, image_count=image_count, size=size, session=session)
    for photo in tqdm(photos, total=image_count, desc=f"Downloading ({size})"):
        filename = _safe_filename(photo.id, size)
        dest_path = os.path.join(output_dir, filename)
        if os.path.exists(dest_path):
            skipped += 1
            continue
        try:
            _download_streaming(session, photo.url, dest_path)
            downloaded += 1
        except Exception as e:
            failed += 1
            try:
                part_path = dest_path + ".part"
                if os.path.exists(part_path):
                    os.remove(part_path)
            except Exception:
                pass
            print(f"Error downloading {photo.id}: {e}")

    print(f"Done. downloaded={downloaded} skipped={skipped} failed={failed} output_dir={output_dir}")

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


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Download images from Pexels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--query", default="African elephant", help="Search query for Pexels")
    parser.add_argument("--size", default="large", choices=["tiny", "small", "medium", "large", "large2x"], help="Image size to download")
    parser.add_argument("--count", type=int, default=2000, help="Number of images to download")
    parser.add_argument("--output-dir", default="images/pexels_elephants", help="Output directory for downloaded images")
    parser.add_argument("--api-key", default=None, help="Pexels API key (uses PEXELS_API_KEY env var if not provided)")
    args = parser.parse_args(argv)

    download_images(
        args.query,
        args.output_dir,
        image_count=args.count,
        size=args.size,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()