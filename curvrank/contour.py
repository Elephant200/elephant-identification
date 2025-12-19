"""
Code to extract contours from images. The most important function is get_contour, which returns the contours and views for an image.
"""
import os
from pprint import pprint
from typing import Literal
import numpy as np
import cv2
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from dotenv import load_dotenv

from utils import resample_polyline, get_files_from_dir


load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
if api_key is None:
    raise ValueError("ROBOFLOW_API_KEY not found in .env file")

# Define Inference clients
CONTOUR_CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key=api_key
)
CONTOUR_CLIENT.configure(InferenceConfiguration(
    confidence_threshold=0.25,
    iou_threshold=0.05
))
CONTOUR_MODEL_URL = "curvrank-contours-zddtc/6"

ANCHOR_CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key=api_key
)
ANCHOR_CLIENT.configure(InferenceConfiguration(
    confidence_threshold=0.05,
    iou_threshold=0.05
))
ANCHOR_MODEL_URL = "anchor-extraction-bwwlq/4"

def get_contour_predictions(image_path: str) -> list[dict]:
    """
    Get the raw coarse contour predictions for an image using the Roboflow API.

    Args:
        image_path (str): The path to the image.

    Returns:
        list[dict]: The coarse contour predictions for the image.
    """
    results = CONTOUR_CLIENT.infer(image_path, model_id=CONTOUR_MODEL_URL)
    predictions = results["predictions"]
    return [
        {
            "points": [(round(point["x"]), round(point["y"])) for point in prediction["points"]],
            "confidence": prediction["confidence"],
        }
        for prediction in predictions
    ]

def get_anchor_points(image_path: str) -> list[dict]:
    """
    Get the anchor points for an image using the Roboflow API.

    Args:
        image_path (str): The path to the image.

    Returns:
        list[dict]: List of dicts containing anchor points for each ear and the confidence of the prediction.
    """
    results = ANCHOR_CLIENT.infer(image_path, model_id=ANCHOR_MODEL_URL)
    anchor_point_pairs = []
    for prediction in results["predictions"]:
        anchor_point_pairs.append({
            "points": [(round(point["x"]), round(point["y"])) for point in prediction["keypoints"]],
            "confidence": prediction["confidence"],
        })
    return anchor_point_pairs

def cut_contours_by_anchors(contours: list[np.ndarray], anchor_point_predictions: list[dict]) -> tuple[list[np.ndarray], list[Literal["left", "right"]]]:
    """
    Cut contours by anchor points and return the longer segments.

    Args:
        contours (list[np.ndarray]): List of contour point arrays. Maximum length is 2.
        anchor_point_predictions (list[dict]): The anchor point predictions.

    Returns:
        list[np.ndarray]: The contours with shorter segments removed.
        list[Literal["left", "right"]]: For each contour, whether it is a left or right ear.
    """
    result = []
    views: list[Literal["left", "right"]] = []
    if len(contours) > len(anchor_point_predictions):
        raise ValueError(f"Insufficient anchor point predictions: {len(contours)} contours but {len(anchor_point_predictions)} anchor point predictions")
    
    for contour_pts in contours:
        contour_centroid = np.mean(contour_pts, axis=0)
        
        # Find closest anchor prediction by comparing anchor centroids to contour centroid
        anchor_centroids = np.array([np.mean(ap["points"], axis=0) for ap in anchor_point_predictions])
        distances_to_centroid = np.linalg.norm(anchor_centroids - contour_centroid, axis=1)
        best_anchor = anchor_point_predictions[np.argmin(distances_to_centroid)]
        
        # Find closest contour point indices for each anchor point
        anchor1, anchor2 = np.array(best_anchor["points"])
        dists1 = np.linalg.norm(contour_pts - anchor1, axis=1)
        dists2 = np.linalg.norm(contour_pts - anchor2, axis=1)
        idx1, idx2 = np.argmin(dists1), np.argmin(dists2)
        
        if ((anchor1 + anchor2) / 2)[0] > contour_centroid[0]:
            views.append("right")
        else:
            views.append("left")

        # Get both possible segments
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        segment_a = contour_pts[idx1:idx2+1]
        segment_b = np.vstack([contour_pts[idx2:], contour_pts[:idx1+1]])
        
        # Keep the longer segment
        result.append(segment_a if len(segment_a) > len(segment_b) else segment_b)
    
    return result, views


def get_contour(image_path: str, view: Literal["side", "front", "auto"] = "auto") -> tuple[list[np.ndarray], list[Literal["left", "right"]]]:
    """
    Get the coarse contour, removing the shorter portion of the ear as defined by the anchor points. Calls the get_contour_predictions and get_anchor_points functions.

    Args:
        image_path (str): The path to the image.
        view (Literal["side", "front", "auto"]): If the view is "side", only one ear will be used. If the view is "front", two ears will be used. If the view is "auto", the view will be determined automatically.

    Returns:
        list[np.ndarray]: The coarse contours for the image. Each is a shape (n, 2) array of contour points.
        list[Literal["left", "right"]]: For each contour, whether it is a left or right ear.
    """
    raw_predictions = get_contour_predictions(image_path)
    anchor_point_predictions = get_anchor_points(image_path)
    
    raw_predictions = [
        {
            "points": np.array(prediction["points"], dtype=np.int32),
            "confidence": prediction["confidence"],
            "area": cv2.contourArea(np.array(prediction["points"], dtype=np.int32))
        }
        for prediction in raw_predictions
    ]
    raw_predictions.sort(key=lambda x: x["area"] * x["confidence"], reverse=True)


    predictions = []
    if len(raw_predictions) == 0:
        raise ValueError("No predictions found")
    elif len(raw_predictions) == 1:
        predictions.append(raw_predictions[0])
    elif view == "side":
        predictions.append(raw_predictions[0])
    elif view == "front":
        predictions.extend(raw_predictions[:2])
    else: # view == "auto"
        pred1 = raw_predictions[0]
        pred2 = raw_predictions[1]

        if pred1["area"] * pred1["confidence"] > 3 * pred2["area"] * pred2["confidence"]: # Much larger
            predictions.append(pred1)
        else:
            predictions.extend([pred1, pred2])
    
    predictions = [pred["points"] for pred in predictions]

    # Extract just the points and resample

    contours = []
    for prediction in predictions:
        resampled = resample_polyline(np.array(prediction), 2).astype(int)
        contours.append(resampled)

    # Cut contours by anchor points
    contours, views = cut_contours_by_anchors(contours, anchor_point_predictions)

    # Convert to lists of tuples
    return [orient_contour(contour) for contour in contours], views

def orient_contour(contour: np.ndarray) -> np.ndarray:
    """
    Orient the contour so that it is going downwards.

    Args:
        contour (np.ndarray): The contour to orient. Shape (n, 2).

    Returns:
        np.ndarray: The oriented contour. Shape (n, 2).
    """
    direction = contour[-1][1] - contour[0][1]
    if direction > 0: # Going downwards
        return contour
    else: # Going upwards
        return contour[::-1]

def visualize_contour(image_path: str):
    predictions = get_contour_predictions(image_path)

    anchor_points = get_anchor_points(image_path)

    base_image = cv2.imread(image_path)
    image = base_image.copy()

    draw_contours(
        image=image,
        contours=predictions,
        color=(255, 0, 0),
        thickness=2,
        draw_points=True
    )
    for pair in anchor_points:
        for anchor_point in pair["points"]:
            cv2.circle(
                img=image,
                center=anchor_point,
                radius=10,
                color=(0, 0, 255),
                thickness=-1
            )
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    predictions, views = get_contour(image_path)

    image = base_image.copy()

    draw_contours(
        image=image,
        contours=predictions,
        color=(0, 255, 0),
        thickness=2,
        draw_points=True
    )

    for contour in predictions:
        cv2.circle(image, contour[0], 10, (255, 255, 255), -1)
        cv2.circle(image, contour[-1], 10, (255, 0, 0), -1)

    cv2.imshow(f"image {', '.join(views)}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_contour_for_image(image_path: str):
    predictions, _ = get_contour(image_path)

    image = cv2.imread(image_path)

    draw_contours(
        image=image,
        contours=predictions,
        color=(0, 255, 0),
        thickness=2,
        draw_points=True
    )

    for contour in predictions: # Line from white to blue
        cv2.circle(image, contour[0], 10, (255, 255, 255), -1)
        cv2.circle(image, contour[-1], 10, (255, 0, 0), -1)

    cv2.imwrite(f"dataset/tough-output/{image_path.split('/')[-1].split('.')[0]}_contour.jpg", image)

if __name__ == "__main__":
    import cv2
    from utils import draw_contours, get_all_images
    import random

    import sys
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    visualize_contour(arg)
                except Exception as e:
                    print(f"Error visualizing contour for image {arg}\nError Message: {e}")
        sys.exit(0)
    
    image_paths = get_all_images("curvrank/preprocessed")

    sample_num = int(input("Enter the number of images to sample, 0 for manual input, or -1 for all: "))
    sample_num = sample_num if sample_num != -1 else len(image_paths)
    image_paths = random.sample(image_paths, sample_num)
    if sample_num == 0:
        image_paths = get_files_from_dir("Please enter the image paths below.")
    
    for image_path in image_paths:
        visualize_contour(image_path)