import os
from pprint import pprint
import random

import cv2
import numpy as np
from dotenv import load_dotenv
from inference.models.sam2 import SegmentAnything2
from inference.core.entities.requests.sam2 import Sam2PromptSet, Sam2Prompt, Box
from inference.core.utils.postprocess import masks2poly
from inference_sdk import InferenceHTTPClient
import supervision as sv

from utils import (
    get_files_from_dir,
    get_int,
    get_list_of_files,
    get_list_of_ints,
    get_multiple_choice,
    is_image,
    print_with_padding,
)

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
if api_key is None:
    raise ValueError("ROBOFLOW_API_KEY not found in .env file")

project_id = "curvrank-contours-zddtc"

model_ids = ["2"]

colors = {
    "1": (255, 0, 0), # Red
    "2": (255, 255, 0), # Yellow
    "3": (0, 255, 0), # Green
    "4": (0, 255, 255), # Cyan
    "5": (0, 0, 255), # Blue
    "6": (255, 0, 255), # Magenta
    "7": (0, 0, 0), # Black
    "8": (255, 255, 255), # White
    "9": (255, 128, 128), # Reddish-Pink
    "10": (128, 255, 128), # Light Green
    "11": (128, 128, 255), # Lavender
}

CLIENT = InferenceHTTPClient(api_url="https://outline.roboflow.com", api_key=api_key)

def get_prediction(client: InferenceHTTPClient, model_id: str, image_path: str) -> list[list[tuple[int, int]]]:
    """
    Given an inference client, a model id, and an image path, return the predictions for the image.

    Args:
        client (InferenceHTTPClient): The inference client.
        model_id (str): The model id.
        image_path (str): The path to the image.

    Returns:
        list[list[tuple[int, int]]]: The predictions for the image, represented as a list of lists of tuples of integers.
    """
    #image = cv2.imread(image_path)
    results = client.infer(image_path, model_id)
    predictions = results["predictions"]
    predictions = [
        {
            "points": [(round(point["x"]), round(point["y"])) for point in prediction["points"]],
            "confidence": prediction["confidence"],
        }
        for prediction in predictions
    ]
    return predictions

def draw_contours(
        image: np.ndarray,
        contours: list[list[tuple[int, int]]],
        contourIdx: int = -1,
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2
    ):
    """
    Wrapper for cv2.drawContours

    Args:
        image (np.ndarray): The image to draw the contours on.
        contours (list[list[tuple[int, int]]]): The contours to draw.

    Returns:
        np.ndarray: The image with the contours drawn on it.
    """
    contours = [np.array(contour["points"]) for contour in contours]
    cv2.drawContours(image, contours, contourIdx, color, thickness)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the image to ensure it is in the correct format for the model.

    Args:
        image (np.ndarray): The image to preprocess.

    Returns:
        np.ndarray: The preprocessed image.
    """
    image = cv2.resize(image, (512, 512))
    return image

while True:
    selection = input("Enter image selection method:\n[1] Random\n[2] Choose (Drag and drop image paths)\n[3] Choose Sample from Directory\n[4] Quit\n")
    if selection == "1":
        base_path = "images/all_elephant_images/"
        image_paths = []
        if get_multiple_choice("Include sheldrick images? (y/n): ", default_choice="Yes") == "Yes":
            if os.path.exists(base_path):
                for folder in os.listdir(base_path):
                    try:
                        for image_path in os.listdir(os.path.join(base_path, folder)):
                            if is_image(image_path):
                                image_paths.append(os.path.join(base_path, folder, image_path))
                    except NotADirectoryError:
                        continue
            else:
                print(f"Directory {base_path} does not exist.")
        if get_multiple_choice("Include ELPephants? (y/n): ", default_choice="Yes") == "Yes":
            elpephants_path = "images/ELPephants"
            if os.path.exists(elpephants_path):
                for image_path in os.listdir(elpephants_path):
                    if is_image(image_path):
                        image_paths.append(f"{elpephants_path}/{image_path}")
            else:
                print(f"Directory {elpephants_path} does not exist.")
        if image_paths:
            random.shuffle(image_paths)
            max_images = len(image_paths)
            requested_images = get_int(f"Enter # images (max {max_images}): ")
            if requested_images > max_images:
                print(f"Only {max_images} images available. Using all available images.")
                requested_images = max_images
            image_paths = image_paths[:requested_images]
        else:
            print("No images found in the specified directories.")
            continue
    elif selection == "2":
        image_paths = get_list_of_files("Please enter the image paths below.")
    elif selection == "3":
        image_paths = get_files_from_dir("Please enter the image paths below.", randomize=True)
    elif selection == "4":
        break
    else:
        print("Invalid selection")
        continue

    if not image_paths:
        print("No images selected. Please try again.")
        continue
    
    valid_image_paths = []
    for image_path in image_paths:
        if not os.path.isfile(image_path):
            print(f"Warning: File not found: {image_path}")
            continue
        if not is_image(image_path):
            print(f"Warning: File is not an image: {image_path}")
            continue
        valid_image_paths.append(image_path)
    
    if len(valid_image_paths) == 0:
        print("No valid image files found. Please try again.")
        continue
    
    image_paths = valid_image_paths
    print(f"Processing {len(image_paths)} images...")

    for i, image_path in enumerate(image_paths):
        print_with_padding(f"{i+1}/{len(image_paths)}")
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        
        image = preprocess_image(image)
        overlay = image.copy()

        x_max = 0
        y_max = 0
        x_min = image.shape[1]
        y_min = image.shape[0]

        for model_id in model_ids:
            print(f"--------------------------------{model_id}--------------------------------")
            print(image_path)
            prediction = get_prediction(CLIENT, f"curvrank-contours-zddtc/{model_id}", overlay)
            pprint(prediction)
            for pred in prediction:
                for point in pred["points"]:
                    x, y = point
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
            draw_contours(overlay, prediction, -1, colors[model_id], 1)
            print(colors[model_id])
        
        # # Use sam2 to segment the image

        # sam2 = SegmentAnything2(model_id="sam2/hiera_large")

        # sam2.embed_image(image)

        # prompts = Sam2PromptSet(prompts=[Sam2Prompt(box=(x_min, y_min, x_max, y_max))])
        # contours = sam2.segment_image(image, prompts=prompts)
        # contours = masks2poly(contours[0])
        # draw_contours(overlay, contours, -1, colors["12"], 2)

        # Resize image to base_width width
        base_width = 1920
        overlay = cv2.resize(overlay, (base_width, int(overlay.shape[0] * base_width / overlay.shape[1])))
        
        cv2.imshow(f"Model Comparison (Image: {image_path.split('/')[-1]}) [{i+1}/{len(image_paths)}]", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
