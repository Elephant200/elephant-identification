import json
import os
import shutil
import time

from inference import get_model
import numpy as np
import supervision as sv
import cv2
from tqdm import tqdm
from typing import Literal

from boxing.get_prediction import get_prediction
from utils import print_with_padding, is_image, clear

project_id = "elephant-identification-research"
model_versions = ["20"]

def get_all_images(dir: str, look_at_subdirs: bool = True) -> list[str]:
  """
  Gets all images in a directory and its subdirectories.

  Args:
    dir (str): The directory to get images from.
    look_at_subdirs (bool): Whether to look at subdirectories.

  Returns:
    list[str]: A list of image paths. Includes the directory path.
  """
  files = []
  for file in os.listdir(dir):
    if not os.path.isdir(os.path.join(dir, file)) and is_image(os.path.join(dir, file)):
      files.append(os.path.join(dir, file))
  if look_at_subdirs:
    for subdir in os.listdir(dir):
      if os.path.isdir(os.path.join(dir, subdir)):
        files.extend(get_all_images(os.path.join(dir, subdir), True))
  return files

def classify_image(detections: sv.Detections) -> Literal["Certain", "Probable", "Uncertain", "Likely Bad", "Rejected"]:
  """
  Classify an image into one of the following categories:
  - Certain
  - Probable
  - Uncertain
  - Likely Bad
  - Rejected

  Args:
    detections (sv.Detections): The detections for the image.

  Returns:
    str: The category of the image.
  """
  def filter(dets, min):
    return [d for d in dets if d["confidence"] > min]
  
  dets = [{
    "class_id": detections.class_id[i],
    "confidence": detections.confidence[i],
    "xyxy": detections.xyxy[i],
    "area": detections.box_area[i],
  } for i in range(len(detections.xyxy))]

  high_conf_dets = filter(dets, 0.8)
  highly_confident_count = len(high_conf_dets)
  if highly_confident_count > 1:
    high_conf_dets.sort(key=lambda x: x["area"], reverse=True)
    ratio = high_conf_dets[0]["area"] / high_conf_dets[1]["area"]
    if ratio >= 4:
      return "Probable"
    elif ratio >= 3:
      return "Uncertain"
    elif ratio >= 2:
      return "Likely Bad"
    else:
      return "Rejected"
  if highly_confident_count == 1:
    if len(filter(dets, 0.9)) == 1:
      return "Certain"
    else:
      return "Probable"

  med_conf_dets = filter(dets, 0.7)
  medium_confident_count = len(med_conf_dets)
  if medium_confident_count > 1:
    med_conf_dets.sort(key=lambda x: x["area"], reverse=True)
    ratio = med_conf_dets[0]["area"] / med_conf_dets[1]["area"]
    if ratio >= 4:
      return "Uncertain"
    elif ratio >= 2:
      return "Likely Bad"
    else:
      return "Rejected"
  if medium_confident_count == 1:
    return "Probable"
  
  if len(filter(dets, 0.6)) == 0:
    return "Rejected"

  return "Uncertain"

def process_and_copy_image(
    image_path: str,
    detections: sv.Detections,
    classification: str,
    unannotated_dirs: dict,
    annotated_dirs: dict,
    trackers: dict,
    box_annotator: sv.BoxAnnotator,
    label_annotator: sv.LabelAnnotator,
    replace_existing: bool = False,
) -> None:
  """
  Process and copy an image to both unannotated and annotated directories.
  
  Args:
    image_path (str): Path to the source image
    detections (sv.Detections): Detection results for annotation
    classification (str): Classification category (Certain, Probable, etc.)
    unannotated_dirs (dict): Dictionary mapping classifications to unannotated directories
    annotated_dirs (dict): Dictionary mapping classifications to annotated directories
    trackers (dict): Dictionary tracking counts for each category
    box_annotator (sv.BoxAnnotator): Box annotator
    label_annotator (sv.LabelAnnotator): Label annotator
    replace_existing (bool): Whether to replace existing annotated images
  """
  image_name = os.path.basename(image_path)
  
  if os.path.exists(os.path.join(unannotated_dirs[classification], image_name)):
    if replace_existing:
      os.remove(os.path.join(unannotated_dirs[classification], image_name))
    else:
      trackers["Already Processed"] += 1
      return

  shutil.copy(image_path, os.path.join(unannotated_dirs[classification], image_name))

  try:
    image = cv2.imread(image_path)
    
    labels = []
    for i in range(len(detections.confidence)):
      confidence = detections.confidence[i]
      area = detections.box_area[i]
      labels.append(f"{round(100 * confidence)}% | {round(area / 1000, 1)}kpx")
    
    annotated_image = box_annotator.annotate(image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(annotated_image, detections=detections, labels=labels)
    
    annotated_path = os.path.join(annotated_dirs[classification], image_name)
    if replace_existing and os.path.exists(annotated_path):
      os.remove(annotated_path)
    cv2.imwrite(annotated_path, annotated_image)
    
    trackers[classification] += 1
  except Exception as e:
    print("Error annotating image: ", e)


if __name__ == "__main__":
  api_key = os.getenv("ROBOFLOW_API_KEY")
  if api_key is None:
    raise ValueError("ROBOFLOW_API_KEY not found in .env file")

  IMAGE_DIR = "/Users/kayoko/Documents/GitHub/elephant-identification/images"
  TYPE = "ELPephants"
  if TYPE == "sheldrick":
    INPUT_DIR = f"{IMAGE_DIR}/all_elephant_images"
  elif TYPE == "ELPephants":
    INPUT_DIR = f"{IMAGE_DIR}/ELPephants"

  # Unannotated directories
  UNANNOTATED_DIRS = {
    "Certain": f"processing/{TYPE}/unannotated/certain",
    "Probable": f"processing/{TYPE}/unannotated/probable",
    "Uncertain": f"processing/{TYPE}/unannotated/uncertain",
    "Likely Bad": f"processing/{TYPE}/unannotated/likely_bad",
    "Rejected": f"processing/{TYPE}/unannotated/rejected",
  }

  # Annotated directories
  ANNOTATED_DIRS = {
    "Certain": f"processing/{TYPE}/annotated/certain",
    "Probable": f"processing/{TYPE}/annotated/probable",
    "Uncertain": f"processing/{TYPE}/annotated/uncertain",
    "Likely Bad": f"processing/{TYPE}/annotated/likely_bad",
    "Rejected": f"processing/{TYPE}/annotated/rejected",
  }

  all_dirs = [*UNANNOTATED_DIRS.values(), *ANNOTATED_DIRS.values()]
  for dir in all_dirs:
    if not os.path.exists(dir):
      os.makedirs(dir)

  models = [get_model(f"{project_id}/{version}", api_key=api_key) for version in tqdm(model_versions, desc="Loading models")]

  # Create lavender annotators
  lavender_color = sv.Color(128, 128, 255)
  box_annotator = sv.BoxAnnotator(color=lavender_color)
  label_annotator = sv.LabelAnnotator(color=lavender_color)

  for model_version, model in zip(model_versions, models):
    trackers = {
      "Certain": 0,
      "Probable": 0,
      "Uncertain": 0,
      "Likely Bad": 0,
      "Rejected": 0,
      "Already Processed": 0,
    }

    print("Unannotated Dirs: ", *UNANNOTATED_DIRS.values())
    print("Annotated Dirs: ", *ANNOTATED_DIRS.values())

    images = get_all_images(INPUT_DIR)

    tq = tqdm(images, desc=f"Processing model v{model_version}")

    start_time = time.time()

    for i, image_path in enumerate(tq):
      image = os.path.basename(image_path)
      if not is_image(image_path):
        continue
      detections = get_prediction(model, image_path)
      classification = classify_image(detections)

      process_and_copy_image(
          image_path=image_path,
          detections=detections,
          classification=classification,
          unannotated_dirs=UNANNOTATED_DIRS,
          annotated_dirs=ANNOTATED_DIRS,
          trackers=trackers,
          box_annotator=box_annotator,
          label_annotator=label_annotator,
          replace_existing=True,
      )

      if i % 100 == 0:
        clear()
        tqdm.write(f"Model v{model_version} - {json.dumps(trackers, indent=2)}")

    clear()
    print_with_padding("FINAL RESULTS")
    print(f"Model v{model_version} - {json.dumps(trackers, indent=2)}")
    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")