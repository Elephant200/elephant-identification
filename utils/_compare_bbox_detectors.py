import os
import random
from pprint import pprint

import cv2
import numpy as np
from dotenv import load_dotenv
from inference import get_model
import supervision as sv
from tqdm import tqdm

from boxing.get_prediction import get_prediction
from dataset_creation.process import classify_image
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

project_id = "elephant-identification-research"
presets = {
    "Best of Each Type": ["13", "16", "20"],
    "Original": ["5", "7", "12", "13", "14", "15", "16"],
    "Less-Augmented": ["15", "16", "19", "20"],
    "New": ["17", "18", "19", "20"],
    "Cyan + New": ["13", "17", "18", "19", "20"],
    "Cyan + Black + New": ["13", "16", "17", "18", "19", "20"],
    "All": ["5", "7", "12", "13", "14", "15", "16", "17", "18", "19", "20"],
}
if get_multiple_choice("Would you like to use a preset group of models? (y/n): ", default_choice="Yes") == "Yes":
    model_versions = presets[get_multiple_choice("Please select a preset. Choices are:\n" + "\n".join(presets.keys()) + "\n", choices=list(presets.keys()))]
else:
    model_versions = get_list_of_ints("Please enter the model versions below.\n")

model_to_color = {
    "5": sv.Color(255, 0, 0), # Red
    "7": sv.Color(255, 255, 0), # Yellow
    "12": sv.Color(0, 255, 0), # Green
    "13": sv.Color(0, 255, 255), # Cyan
    "14": sv.Color(0, 0, 255), # Blue
    "15": sv.Color(255, 0, 255), # Magenta
    "16": sv.Color(0, 0, 0), # Black
    "17": sv.Color(255, 255, 255), # White
    "18": sv.Color(255, 128, 128), # Reddish-Pink
    "19": sv.Color(128, 255, 128), # Light Green
    "20": sv.Color(128, 128, 255), # Lavender
}

def load_models(model_versions: list[str | int]) -> tuple[list, list[sv.Color], list[tuple[sv.BoxAnnotator, sv.LabelAnnotator]]]:
    """
    Get the models and colors for a list of model versions.

    Args:
        model_versions (list[str | int]): List of model versions.

    Returns:
        tuple[list[Model], list[sv.Color], tuple[sv.BoxAnnotator, sv.LabelAnnotator]]: Tuple of lists of models and colors.
    """
    models = [
        get_model(f"{project_id}/{version}", api_key=api_key)
        for version in tqdm(model_versions, desc="Loading models")
    ]
    colors = [model_to_color[str(version)] for version in model_versions]
    annotators = [(sv.BoxAnnotator(color=color), sv.LabelAnnotator(color=color)) for color in colors]
    return models, colors, annotators

def generate_colors(n: int) -> list[sv.Color]:
    """
    Generate a list of colors for n models.

    Args:
        n (int): Number of models.

    Returns:
        list[sv.Color]: List of colors.
    """
    def hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
        h = hue / 60
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)
        return r, g, b

    colors = []
    for i in range(n):
        h = (i * 360) // n
        s = 1.0
        v = 1.0
        colors.append(sv.Color(*hsv_to_rgb(h, s, v)))
    
    return colors

models, colors, annotators = load_models(model_versions)

while True:
    selection = input("Enter image selection method:\n[1] Random\n[2] Choose (Drag and drop image paths)\n[3] Choose Sample from Directory\n[4] Edit model versions\n[5] Quit\n")
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
        if get_multiple_choice("Would you like to use a preset group of models? (y/n): ", default_choice="Yes") == "Yes":
            model_versions = presets[get_multiple_choice("Please select a preset. Choices are:\n" + "\n".join([f"{i+1}. {preset}" for i, preset in enumerate(presets.keys())]), choices=list(presets.keys()))]
        else:
            model_versions = get_list_of_ints("Please enter the model versions below.\n")
        models, colors, annotators = load_models(model_versions)
        continue
    elif selection == "5":
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

        overlay = image.copy()

        for model, (box_annotator, label_annotator), version in zip(models, annotators, model_versions):
            detections = get_prediction(model, image_path)
            classification = classify_image(detections)
            detection = [d for d in detections]
            print(detections.xyxy)
            print(detections.class_id)
            print(f"--------------------------------{version}--------------------------------")
            pprint(detection)
            labels = [f"v{version} {round(d[2] * 100)}%" for d in detections]
            overlay = box_annotator.annotate(scene=overlay, detections=detections)
            overlay = label_annotator.annotate(scene=overlay, detections=detections, labels=labels)

        # Resize image to base_width width
        base_width = 1920
        overlay = cv2.resize(overlay, (base_width, int(overlay.shape[0] * base_width / overlay.shape[1])))
        
        legend_height = 30 * len(model_versions) + 20
        legend_width = 200
        legend_x = overlay.shape[1] + 10
        legend_y = 10

        overlay = np.pad(overlay, ((0, 0), (0, legend_width + 20), (0, 0)), mode="constant", constant_values=255)
        
        print(overlay.shape, legend_x, legend_y, legend_width, legend_height, sep="\n")

        cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (255, 255, 255), -1)
        cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (0, 0, 0), 2)

        cv2.putText(overlay, "Model Versions", (legend_x + 10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for j, (version, color) in enumerate(zip(model_versions, colors)):
            y_pos = legend_y + 40 + j * 25
            cv2.rectangle(overlay, (legend_x + 10, y_pos - 10), (legend_x + 30, y_pos + 10), (color.b, color.g, color.r), -1)
            cv2.putText(overlay, f"v{version}", (legend_x + 40, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow(f"Model Comparison (Classification: {classification}) (Image: {image_path.split('/')[-1]}) [{i+1}/{len(image_paths)}]", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
