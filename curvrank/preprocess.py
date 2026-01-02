import numpy as np
from typing import Literal
import os
import cv2
from tqdm import tqdm
from curvrank.contour import get_contour
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
import supervision as sv
from pycocotools import mask as mask_util

from utils import get_list_of_files

load_dotenv()

def clamp(x, lower, upper):
    """Equivalent to max(lower, min(x, upper))"""
    return max(lower, min(x, upper))

TARGET_SIZE = 432
PADDING = 0.10 # 10% padding around the initial contour

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

def remove_background(image: np.ndarray) -> np.ndarray:
    """
    Remove the background from the image using SAM3
    """
    body_preds = CLIENT.run_workflow(
        workspace_name="elephantidentificationresearch",
        workflow_id="sam3-with-prompts",
        images={
            "image": image
        },
        parameters={
            "prompts": "elephant"
        },
        use_cache=True
    )
    body_preds = body_preds[0]["sam"]["predictions"]

    tusk_preds = CLIENT.run_workflow(
        workspace_name="elephantidentificationresearch",
        workflow_id="sam3-with-prompts",
        images={
            "image": image
        },
        parameters={
            "prompts": "tusk"
        },
        use_cache=True
    )
    tusk_preds = tusk_preds[0]["sam"]["predictions"]

    def process_pred(pred: dict) -> dict:
        return {
            **pred,
            "mask": mask_util.decode(pred["rle_mask"])
        }

    body_preds = [process_pred(pred) for pred in body_preds]
    tusk_preds = [process_pred(pred) for pred in tusk_preds]

    # For each body prediction, find overlapping tusks and merge them into the body mask
    for body_pred in body_preds:
        body_mask = body_pred["mask"]
        for tusk_pred in tusk_preds:
            tusk_mask = tusk_pred["mask"]
            overlap = np.logical_and(body_mask, tusk_mask).sum()
            if overlap > 0:
                body_pred["mask"] = np.logical_or(body_mask, tusk_mask).astype(np.uint8)
                body_mask = body_pred["mask"]

    # Combine all body masks into a single mask
    if body_preds:
        combined_mask = body_preds[0]["mask"].copy()
        for body_pred in body_preds[1:]:
            combined_mask = np.logical_or(combined_mask, body_pred["mask"])
        combined_mask = combined_mask.astype(np.uint8)
    else:
        combined_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Apply mask to image (set background to black)
    result = cv2.bitwise_and(image, image, mask=combined_mask)
    return result

def preprocess_images(
        image_paths: list[str],
        output_dir: str = "curvrank/preprocessed",
        force: bool = False
    ) -> tuple[list[str], list[Literal["left", "right"]], list[str]]:
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

    out_paths: list[str] = []
    out_views: list[Literal["left", "right"]] = []
    out_names: list[str] = []

    if not force:
        existing_images = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
        image_paths = [ip for ip in image_paths if f"{ip.split('/')[-1].split('.')[0]}_right.jpg" not in existing_images and f"{ip.split('/')[-1].split('.')[0]}_left.jpg" not in existing_images]
        if len(image_paths) == 0:
            print("No images to preprocess")
        print(f"Skipping {len(existing_images)} images that already exist")
        for ip in existing_images:
            out_paths.append(os.path.join(output_dir, ip))
            out_views.append(ip.split(".")[0].split("_")[-1])
            out_names.append("_".join(ip.split(".")[0].split("_")[:-1]))

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

def get_contours(image_paths: list[str], force: bool = False):
    all_contours = []
    for image_path in image_paths:
        try:
            contours, views = get_contour(image_path)
        except Exception as e:
            print(f"Error getting contours for image {image_path}\nError Message: {e}")
            continue
        all_contours.extend(contours)
    return all_contours

if __name__ == "__main__":
    for image in get_list_of_files("Enter the path to the images to preprocess: "):
        image = cv2.imread(image)
        image = remove_background(image)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #preprocess_images([image], output_dir="curvrank/preprocessed")