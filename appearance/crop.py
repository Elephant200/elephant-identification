"""Face detection and background removal for elephant images."""
import json
import os
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from pycocotools import mask as mask_util

load_dotenv()

FACE_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)
FACE_MODEL_ID = "elephant-identification-research/20"

BG_CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

FACE_CACHE_DIR = os.path.join("cache", "appearance", "crop")
SAM_CACHE_DIR = os.path.join("cache", "appearance", "sam")


def get_face_cache_path(image_path: str) -> str:
    """Convert an image path to a cache path in cache/appearance/crop/filename.json."""
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    return os.path.join(FACE_CACHE_DIR, name_without_ext + ".json")


def load_cached_face(cache_path: str) -> list[dict] | None:
    """Load face predictions from cache if exists."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
        return data["predictions"]
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def save_face_to_cache(cache_path: str, predictions: list[dict]) -> None:
    """Save face predictions to cache JSON."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    data = {"predictions": predictions}
    with open(cache_path, "w") as f:
        json.dump(data, f)


def detect_face(image_path: str, use_cache: bool = True) -> list[dict]:
    """Detect elephant faces in an image using YOLOv11.
    
    Args:
        image_path: Path to the image file.
        use_cache: Whether to use cached results if available.
        
    Returns:
        List of face prediction dicts with x, y, width, height, confidence.
    """
    cache_path = get_face_cache_path(image_path)
    
    if use_cache:
        cached = load_cached_face(cache_path)
        if cached is not None:
            return cached
    
    result = FACE_CLIENT.infer(image_path, model_id=FACE_MODEL_ID)
    predictions = result.get("predictions", [])
    
    save_face_to_cache(cache_path, predictions)
    return predictions


def get_sam_cache_path(image_path: str) -> str:
    """Convert an image path to a SAM cache path in cache/appearance/sam/filename.json."""
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    return os.path.join(SAM_CACHE_DIR, name_without_ext + ".json")


def load_cached_sam(cache_path: str) -> dict | None:
    """Load SAM results from cache if exists."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return None


def save_sam_to_cache(cache_path: str, payload: dict) -> None:
    """Save SAM results to cache JSON."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(payload, f)


def _extract_rles_from_response(response: list[dict], key: str) -> list[dict]:
    """Extract RLE masks from SAM3 workflow response."""
    if not response:
        return []
    preds = response[0].get(key, {}).get("predictions", [])
    return [p["rle_mask"] for p in preds if "rle_mask" in p]


def smooth_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Keep largest connected component and close small holes."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _run_sam_workflow(image: np.ndarray) -> list[dict]:
    """Run SAM3 workflow."""
    return BG_CLIENT.run_workflow(
        workspace_name="elephantidentificationresearch",
        workflow_id="sam3-with-prompts",
        images={"image": image},
        use_cache=True
    )


def remove_background(
    image: np.ndarray,
    image_path: str | None = None,
    crop_bbox: Any | None = None
) -> np.ndarray:
    """Remove the background from the image using SAM3.
    
    Uses RLE-level operations for efficiency, only decoding once at the end.
    
    Args:
        image: The cropped face image as numpy array.
        image_path: Original image path (for caching). If None, no caching.
        crop_bbox: Bounding box of crop in original image coords. Stored in cache.
        
    Returns:
        Image with background removed.
    """
    body_rles: list[dict] = []
    tusk_rles: list[dict] = []
    
    cache_path = get_sam_cache_path(image_path) if image_path else None
    
    if cache_path:
        cached = load_cached_sam(cache_path)
        if cached is not None:
            body_rles = [p["rle_mask"] for p in cached.get("sam_body", {}).get("predictions", [])]
            tusk_rles = [p["rle_mask"] for p in cached.get("sam_tusk", {}).get("predictions", [])]
    
    if not body_rles:
        response = _run_sam_workflow(image)
        body_rles = _extract_rles_from_response(response, "sam_body")
        tusk_rles = _extract_rles_from_response(response, "sam_tusk")
        
        if cache_path:
            h, w = image.shape[:2]
            cache_payload = {
                "crop_bbox": crop_bbox,
                "image_size": {"width": w, "height": h},
                "sam_body": {
                    "predictions": [{"rle_mask": rle} for rle in body_rles]
                },
                "sam_tusk": {
                    "predictions": [{"rle_mask": rle} for rle in tusk_rles]
                }
            }
            save_sam_to_cache(cache_path, cache_payload)
    
    for i, body_rle in enumerate(body_rles):
        for tusk_rle in tusk_rles:
            iou = mask_util.iou([body_rle], [tusk_rle], [0])
            if iou[0, 0] > 0:
                body_rles[i] = mask_util.merge([body_rle, tusk_rle], intersect=0)
                body_rle = body_rles[i]

    if body_rles:
        combined_rle = mask_util.merge(body_rles, intersect=0)
        combined_mask = mask_util.decode(combined_rle)
        combined_mask = smooth_mask(combined_mask)
    else:
        combined_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    result = cv2.bitwise_and(image, image, mask=combined_mask)
    return result
