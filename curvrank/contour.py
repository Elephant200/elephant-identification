"""
Code to extract contours from images
"""
import os
from pprint import pprint
import numpy as np
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from dotenv import load_dotenv


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
    confidence_threshold=0.50,
    iou_threshold=0.05
))
CONTOUR_MODEL_URL = "curvrank-contours-zddtc/6"

ANCHOR_CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key=api_key
)
ANCHOR_CLIENT.configure(InferenceConfiguration(
    confidence_threshold=0.25,
    iou_threshold=0.25
))
ANCHOR_MODEL_URL = "anchor-extraction-bwwlq/4"

def get_coarse_contour_predictions(image_path: str) -> list[dict]:
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

def get_anchor_points(image_path: str) -> list[tuple[int, int]]:
    """
    Get the anchor points for an image using the Roboflow API.
    """
    results = ANCHOR_CLIENT.infer(image_path, model_id=ANCHOR_MODEL_URL)
    predictions = results["predictions"]
    return predictions

if __name__ == "__main__":
    image_path = "curvrank/ex_image.jpg"
    predictions = get_coarse_contour_predictions(image_path)
    pprint(predictions)

    anchor_points = get_anchor_points(image_path)
    pprint(anchor_points)

    import cv2
    from utils import draw_contours
    image = cv2.imread(image_path)
    
    draw_contours(
        image=image,
        contours=predictions,
        color=(0, 0, 255),
        thickness=2,
        draw_points=True
    )
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()