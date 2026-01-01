import os
from typing import Any

import cv2
import numpy as np
import supervision as sv
import inference
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")


def get_bbox(model: str | Any, image: str | np.ndarray) -> sv.Detections:
    """Get bounding box predictions for an image using a model."""
    if isinstance(model, str):
        model = inference.get_model(f"elephant-identification-research/{model}", api_key=api_key)
    if isinstance(image, str):
        image = cv2.imread(image)
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)
    return detections


if __name__ == "__main__":
    if api_key is None:
        raise ValueError("ROBOFLOW_API_KEY not found in .env file")

    model_id = "20"
    image_path = "images/all_elephant_images/ahmed/ahmed_8.jpg"
    print(get_bbox(model_id, image_path))
