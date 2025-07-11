"""
This script uses a consensus method with the Roboflow API to determine if an image definitively contains an elephant.
"""
import logging
import os
from pprint import pprint
from typing import Dict
from itertools import combinations

import cv2
from dotenv import load_dotenv
import inference
import supervision as sv
from tqdm import tqdm

from utility import print_with_padding, pad_with_char

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (tuple): (x1, y1, x2, y2)
        box2 (tuple): (x1, y1, x2, y2)

    Returns:
        float: IoU value between 0.0 and 1.0
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def check_elephant(models: dict, image_path: str) -> int:
    """
    Determines if exactly one elephant is confidently present in the image,
    using ensemble agreement from multiple Roboflow models.

    Args:
        models (dict): Dictionary of Roboflow models mapping version to model.
        image_path (str): Path to the input image.

    Returns:
        int: Agreement score (out of 10) if a single elephant is confidently present, -1 otherwise.
    """
    agreement_score = 0.0

    model_weights = { # Based on model accuracy
        "13": 1.0,
        "16": 0.95,
        "15": 0.4,
        "14": 0.3,
        "12": 0.2
    }

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    detections_dict: Dict[str, sv.Detections] = {}
    for version, model in models.items():
        results = model.infer(image)[0]
        detections = sv.Detections.from_inference(results)
        detections_dict[version] = detections

    bboxes = {}
    multi_elephant_flags = 0

    for version, detections in detections_dict.items():
        high_conf_detections = [d for d in detections if d[2] >= 0.5]
        if len(high_conf_detections) >= 2:
            multi_elephant_flags += 1
        if high_conf_detections:
            top_det = high_conf_detections[0][0]
            x1, y1, x2, y2 = float(top_det[0]), float(top_det[1]), float(top_det[2]), float(top_det[3])
            bboxes[version] = (x1, y1, x2, y2)
    #pprint(bboxes)

    agreement_score -= multi_elephant_flags * 0.2
    logger.info(f"Multi-elephant flags: {multi_elephant_flags}")

    if len(bboxes) < 3:
        logger.info("Less than 3 models detected elephants - rejected")
        return -1

    for (v1, box1), (v2, box2) in combinations(bboxes.items(), 2):
        iou = compute_iou(box1, box2)
        weight = round((model_weights[v1] + model_weights[v2]) / 2, 2)
        logger.info(f"Between {v1} and {v2}: IOU: {iou} \tWeight: {weight}\tScore: {iou * weight}")
        agreement_score += iou * weight

    return round(agreement_score / (sum(model_weights.values()) * 2), 4)

def test_images(models: dict):
    test_image_paths = [
        "images/all_elephant_images/ahmed/ahmed_0.jpg", # Obvious
        "images/all_elephant_images/ahmed/ahmed_1.jpg", # Not obvious, but there
        "images/all_elephant_images/ahmed/ahmed_2.jpg", # No elephant
        "images/all_elephant_images/ahmed/ahmed_7.jpg", # Obvious
        "images/all_elephant_images/ahmed/ahmed_8.jpg", # Two elephants
        "images/all_elephant_images/ahmed/ahmed_9.jpg", # Obscured with many elephants
        "images/all_elephant_images/ahmed/ahmed_10.jpg", # Obscured with many elephants
        "images/all_elephant_images/ahmed/ahmed_11.jpg", # Obvious
        "images/all_elephant_images/aitong/aitong_0.jpg", # Obvious
        "images/all_elephant_images/aitong/aitong_1.jpg", # Two elephants
        "images/all_elephant_images/aitong/aitong_2.jpg", # Two elephants
        "images/all_elephant_images/aitong/aitong_3.jpg", # Obvious
        "images/all_elephant_images/aitong/aitong_4.jpg", # Many elephants
        "images/all_elephant_images/aitong/aitong_5.jpg", # Two elephants
        "images/all_elephant_images/ajali/ajali_0.jpg",
        "images/all_elephant_images/ajali/ajali_1.jpg",
        "images/all_elephant_images/ajali/ajali_2.jpg",
        "images/all_elephant_images/ajali/ajali_3.jpg",
        "images/all_elephant_images/ajali/ajali_4.jpg",
        "images/all_elephant_images/ajali/ajali_5.jpg",
        "images/all_elephant_images/ajali/ajali_6.jpg"
    ]

    for image_path in test_image_paths:
        logger.info(pad_with_char(f"Checking {image_path}"))
        result = check_elephant(models=models, image_path=image_path)
        logger.info(f"Result: {result}")

        THRESHOLD = 0.7
        logger.info(f"Single elephant detected" if result > THRESHOLD else "No single elephant detected")

def sort_through_sheldrick(models: dict, threshold: float = 0.7):
    """
    Iterate through all sheldrick images and check if they contain an elephant.
    """
    base_path = "images/all_elephant_images/"
    image_paths = []
    for folder in os.listdir(base_path):
        # Only include folders that are alphabetically before Kainuk
        if folder < "kainuk":
            logger.info(folder)
            try:
                for image_path in os.listdir(os.path.join(base_path, folder)):
                    image_paths.append(os.path.join(base_path, folder, image_path))
            except NotADirectoryError:
                continue
            

    logger.info(f"Found {len(image_paths)} images")

    ok_list = []
    for image_path in tqdm(image_paths, desc=f"Checking images for {threshold}"):
        logger.info(pad_with_char(f"Checking {image_path}"))
        result = check_elephant(models=models, image_path=image_path)
        logger.info(f"Result: {result}")

        if result > threshold:
            ok_list.append(image_path)
            logger.info("Single elephant detected")
        else:
            logger.info("No single elephant detected")

    logger.info(f"Found {len(ok_list)} images with a single elephant")
    logger.info(ok_list)
    
    # Compare accuracy with ground truth
    # Images with a single elephant are in the elephant_images folder with the same name as the orphan

    tp = 0
    for image_path in ok_list:
        image_path = image_path.replace("/all_elephant_images/", "/elephant_images/")
        if os.path.exists(image_path):
            tp += 1
            logger.info(f"Image {image_path} exists")
        else:
            logger.info(f"Image {image_path} does not exist")
    
    fp = len(ok_list) - tp

    fn = 0
    for image_path in [path for path in image_paths if path not in ok_list]:
        if os.path.exists(image_path):
            fn += 1

    tn = len(image_paths) - tp - fp - fn

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    print_with_padding(f"Evaluation (Threshold: {threshold})")
    print(f"{'Precision':<10}: {tp:<7} / {tp + fp:<11} -> {round(precision * 100, 2):>6}%")
    print(f"{'Recall':<10}: {tp:<7} / {tp + fn:<11} -> {round(recall * 100, 2):>6}%")
    print(f"{'F1 Score':<10}: {round(2 * precision * recall, 4):<7} / {round(precision + recall, 4):<11} -> {round(f1 * 100, 2):>6}%")
    return f1


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if api_key is None:
        raise ValueError("ROBOFLOW_API_KEY not found in .env file")

    project_id = "elephant-identification-research"
    model_versions = ["12", "13", "14", "15", "16"]

    models = {
        version: inference.get_model(f"{project_id}/{version}", api_key=api_key)
        for version in tqdm(model_versions, desc="Loading models")
    }

    # results = {}
    
    for i in range(15):
        try:
            print(check_elephant(models, f"/Users/kayoko/Documents/GitHub/elephant-identification/images/elephant_images/kilulu/kilulu_{i}.jpg"))
        except Exception as e:
            print(e)

    # for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    #     f1 = sort_through_sheldrick(models, threshold)
    #     print(f"Threshold: {threshold}\tF1 Score: {f1}")
    #     results[threshold] = f1

    # print(results)

"""
RESULTS OF RUNNING THE SCRIPT:

Checking images for 0.5: 100%|██████████████████████████████████████████████████████████████| 642/642 [00:30<00:00, 20.95it/s]
=================================================Evaluation (Threshold: 0.5)=================================================
Precision : 267     / 318         ->  83.96%
Recall    : 267     / 591         ->  45.18%
Accuracy  : 267     / 642         ->  41.59%
F1 Score  : 0.7586  / 1.2914      ->  58.75%
Threshold: 0.5  F1 Score: 0.5874587458745875
Checking images for 0.55: 100%|█████████████████████████████████████████████████████████████| 642/642 [00:30<00:00, 20.94it/s]
=================================================Evaluation (Threshold: 0.55)=================================================
Precision : 240     / 280         ->  85.71%
Recall    : 240     / 602         ->  39.87%
Accuracy  : 240     / 642         ->  37.38%
F1 Score  : 0.6834  / 1.2558      ->  54.42%
Threshold: 0.55 F1 Score: 0.5442176870748299
Checking images for 0.6: 100%|██████████████████████████████████████████████████████████████| 642/642 [00:31<00:00, 20.57it/s]
=================================================Evaluation (Threshold: 0.6)=================================================
Precision : 203     / 238         ->  85.29%
Recall    : 203     / 607         ->  33.44%
Accuracy  : 203     / 642         ->  31.62%
F1 Score  : 0.5705  / 1.1874      ->  48.05%
Threshold: 0.6  F1 Score: 0.4804733727810651
Checking images for 0.65: 100%|█████████████████████████████████████████████████████████████| 642/642 [00:31<00:00, 20.64it/s]
=================================================Evaluation (Threshold: 0.65)=================================================
Precision : 203     / 238         ->  85.29%
Recall    : 203     / 607         ->  33.44%
Accuracy  : 203     / 642         ->  31.62%
F1 Score  : 0.5705  / 1.1874      ->  48.05%
Threshold: 0.65 F1 Score: 0.4804733727810651
Checking images for 0.7: 100%|██████████████████████████████████████████████████████████████| 642/642 [00:30<00:00, 20.73it/s]
=================================================Evaluation (Threshold: 0.7)=================================================
Precision : 198     / 231         ->  85.71%
Recall    : 198     / 609         ->  32.51%
Accuracy  : 198     / 642         ->  30.84%
F1 Score  : 0.5574  / 1.1823      ->  47.14%
Threshold: 0.7  F1 Score: 0.4714285714285714
Checking images for 0.75: 100%|█████████████████████████████████████████████████████████████| 642/642 [00:30<00:00, 20.77it/s]
=================================================Evaluation (Threshold: 0.75)=================================================
Precision : 191     / 223         ->  85.65%
Recall    : 191     / 610         ->  31.31%
Accuracy  : 191     / 642         ->  29.75%
F1 Score  : 0.5364  / 1.1696      ->  45.86%
Threshold: 0.75 F1 Score: 0.4585834333733492
Checking images for 0.8: 100%|██████████████████████████████████████████████████████████████| 642/642 [00:30<00:00, 21.00it/s]
=================================================Evaluation (Threshold: 0.8)=================================================
Precision : 176     / 205         ->  85.85%
Recall    : 176     / 613         ->  28.71%
Accuracy  : 176     / 642         ->  27.41%
F1 Score  : 0.493   / 1.1456      ->  43.03%
Threshold: 0.8  F1 Score: 0.43031784841075793
Checking images for 0.85: 100%|█████████████████████████████████████████████████████████████| 642/642 [00:30<00:00, 21.04it/s]
=================================================Evaluation (Threshold: 0.85)=================================================
Precision : 152     / 174         ->  87.36%
Recall    : 152     / 620         ->  24.52%
Accuracy  : 152     / 642         ->  23.68%
F1 Score  : 0.4283  / 1.1187      ->  38.29%
Threshold: 0.85 F1 Score: 0.38287153652392947
Checking images for 0.9: 100%|██████████████████████████████████████████████████████████████| 642/642 [00:32<00:00, 19.57it/s]
=================================================Evaluation (Threshold: 0.9)=================================================
Precision : 106     / 119         ->  89.08%
Recall    : 106     / 629         ->  16.85%
Accuracy  : 106     / 642         ->  16.51%
F1 Score  : 0.3002  / 1.0593      ->  28.34%
Threshold: 0.9  F1 Score: 0.2834224598930481
Checking images for 0.95: 100%|█████████████████████████████████████████████████████████████| 642/642 [00:31<00:00, 20.11it/s]
=================================================Evaluation (Threshold: 0.95)=================================================
Precision : 20      / 25          ->   80.0%
Recall    : 20      / 637         ->   3.14%
Accuracy  : 20      / 642         ->   3.12%
F1 Score  : 0.0502  / 0.8314      ->   6.04%
Threshold: 0.95 F1 Score: 0.060422960725075525
Checking images for 1.0: 100%|██████████████████████████████████████████████████████████████| 642/642 [00:30<00:00, 20.76it/s]
"""