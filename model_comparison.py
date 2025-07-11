import os
import random
import cv2
import numpy as np
from dotenv import load_dotenv
from inference import get_model
import supervision as sv
from tqdm import tqdm
from pprint import pprint
from utility import pad_with_char

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
if api_key is None:
    raise ValueError("ROBOFLOW_API_KEY not found in .env file")

project_id = "elephant-identification-research"
model_versions = ["12", "13", "14", "15", "16"]
model_to_color = {
    "5": sv.Color(255, 0, 0),
    "7": sv.Color(255, 255, 0),
    "12": sv.Color(0, 255, 0),
    "13": sv.Color(0, 255, 255),
    "14": sv.Color(0, 0, 255),
    "15": sv.Color(255, 0, 255),
    "16": sv.Color(0, 0, 0),
}
colors = [model_to_color[version] for version in model_versions]

models = [
    get_model(f"{project_id}/{version}", api_key=api_key)
    for version in tqdm(model_versions, desc="Loading models")
]

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

annotators = [(sv.BoxAnnotator(color=color), sv.LabelAnnotator(color=color)) for color in colors]

while True:
    selection = input("Enter image selection method:\n[1] Random\n[2] Choose\n[3] Quit\n")
    if selection == "1":
        base_path = "images/all_elephant_images/"
        image_paths = []
        if input("Include sheldrick images? (y/n): ").lower() != "n":
            for folder in os.listdir(base_path):
                try:
                    for image_path in os.listdir(os.path.join(base_path, folder)):
                        image_paths.append(os.path.join(base_path, folder, image_path))
                except NotADirectoryError:
                    continue
        if input("Include ELPephants? (y/n): ").lower() != "n":
            for image_path in os.listdir("images/ELPephants"):
                image_paths.append(f"images/ELPephants/{image_path}")
        random.shuffle(image_paths)
        image_paths = image_paths[:int(input("Enter # images: "))]
    elif selection == "2":
        image_paths = input("Enter image paths:")[1:-1].split("''")
    elif selection == "3":
        break
    else:
        print("Invalid selection")
        continue

    for i, image_path in enumerate(image_paths):
        print(f"--------------------------{i+1}/{len(image_paths)}---------------------------")
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        overlay = image.copy()

        for model, (box_annotator, label_annotator), version in zip(models, annotators, model_versions):
            results = model.infer(image)[0]
            detections = sv.Detections.from_inference(results)
            detection = [d for d in detections]
            print(detections.xyxy)
            print(f"--------------------------------{version}--------------------------------")
            pprint(detection)
            labels = [f"v{version} {d[2]:.2f}%" for d in detections]
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

        cv2.imshow(f"Model Comparison (Image: {image_path.split('/')[-1]}) [{i+1}/{len(image_paths)}]", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
