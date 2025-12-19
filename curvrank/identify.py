import os
import numpy as np
import cv2
from tqdm import tqdm
from curvrank.contour import get_contour
from curvrank.curve import curvature, curvature_descriptors
from typing import Literal

from utils import get_all_images

TARGET_SIZE = 432
PADDING = 0.10 # 10% padding around the initial contour

def clamp(x, lower, upper):
    """Equivalent to max(lower, min(x, upper))"""
    return max(lower, min(x, upper))

def preprocess_images(image_paths: list[str], output_dir: str = "curvrank/preprocessed", force: bool = False) -> tuple[list[str], list[Literal["left", "right"]], list[str]]:
    """
    Preprocess the images to prepare for contour extraction, descriptor calculation, and LNBNN matching.
    For every image, split it into different images, one for each ear. Return the list of images, the views, and the names corresponding to these new images of the ears. Each image is resized to 432x432 for rf-detr inference.

    Args:
        image_paths (list[str]): List of paths to the images. Extracts names from the paths.
        output_dir (str): Directory to save the preprocessed images.
        force (bool): Whether to force reprocessing of images that already exist.

    Returns:
        tuple[list[str], list[Literal["left", "right"]], list[str]]: List of paths to the new images, the views, and the names corresponding to these new images of the ears.
    """
    failed_images: list[str] = []
    os.makedirs(output_dir, exist_ok=True)

    if not force:
        existing_images = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
        image_paths = [ip for ip in image_paths if f"{ip.split('/')[-1].split('.')[0]}_right.jpg" not in existing_images and f"{ip.split('/')[-1].split('.')[0]}_left.jpg" not in existing_images]
        if len(image_paths) == 0:
            print("No images to preprocess")

        print(f"Skipping {len(existing_images)} images that already exist")

    out_paths: list[str] = []
    out_views: list[Literal["left", "right"]] = []
    out_names: list[str] = []

    for image_path in tqdm(image_paths, desc="Preprocessing images"):
        name = image_path.split("/")[-1].split("_")[0]
        # assert name.isnumeric(), f"Could not extract name from filepath: {image_path}"
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        try:
            contours, views = get_contour(image_path)
        except Exception as e:
            print(f"Error getting contour for image {image_path}\nError Message: {e}")
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
                print(f"Error preprocessing image {image_path}\nData:\nContour: {contour}\nView: {view}\nBounding Box: {x_min, y_min, x_max, y_max}\nError Message: {e}")
                failed_images.append(image_path)
                continue

    return out_paths, out_views, out_names

def pipeline(image_paths: list[str]) -> list[tuple[str, list[np.ndarray]]]:
    """
    Pipeline to perform contour extraction, descriptor calculation, and LNBNN matching.
    """
    image_paths, views, names = preprocess_images(image_paths, output_dir="curvrank/preprocessed")

if __name__ == "__main__":
    image_paths = get_all_images("processing/ELPephants/unannotated/certain")
    preprocess_images(image_paths, output_dir="curvrank/preprocessed")

    # for name, descriptors in pipeline(image_paths, views, names):
    #     print(name, descriptors)