import inference
import supervision as sv
import cv2
from dotenv import load_dotenv
import os
import numpy as np
from typing import Any

def get_prediction(model: str | Any, image: str | np.ndarray) -> sv.Detections:
    """
    Get the predictions for an image using a model.

    Args:
        model (str | Any): The model to use. If a string, it is assumed to be the model version.
        image (str | np.ndarray): The image to predict. If a string, it is assumed to be the path to the image. If a numpy array, it is assumed to be the image.

    Returns:
    """
    if isinstance(model, str):
        model = inference.get_model(f"elephant-identification-research/{model}", api_key=api_key)
    if isinstance(image, str):
        image = cv2.imread(image)
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)
    return detections

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if api_key is None:
        raise ValueError("ROBOFLOW_API_KEY not found in .env file")

    model_id = "20"
    image_path = "images/all_elephant_images/ahmed/ahmed_8.jpg"
    print(get_prediction(model_id, image_path))