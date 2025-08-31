from pprint import pprint
import pandas as pd
import os
import shutil
from utils import is_image

def extract_name_from_filepath(filepath: str) -> str:
    filepath = os.path.basename(filepath)
    return filepath.split("_")[0]

# Move images to desired location from /processing/ELPephants/unannotated/certain
images: list[dict] = []

root_dir = "/Users/kayoko/Documents/GitHub/elephant-identification"

TYPES = ["ELPephants"]
FORCE = True

for TYPE in TYPES:
    if not os.path.exists(f"{root_dir}/dataset/{TYPE}"):
        os.makedirs(f"{root_dir}/dataset/{TYPE}")

    if FORCE:
        shutil.rmtree(f"{root_dir}/dataset/{TYPE}")
        os.makedirs(f"{root_dir}/dataset/{TYPE}")

    for image in os.listdir(f"{root_dir}/processing/{TYPE}/cropped/certain"):
        if not is_image(f"{root_dir}/processing/{TYPE}/cropped/certain/{image}"):
            continue
        try:
            source_path = f"{root_dir}/processing/{TYPE}/cropped/certain/{image}"
            dest_path = f"{root_dir}/dataset/{TYPE}/{image}"
            # print(f"Copying image from {source_path} to {dest_path}")
            shutil.copy2(source_path, dest_path)
            images.append({"name": extract_name_from_filepath(image), "filepath": dest_path, "data_source": TYPE})
        except Exception as e:
            print(f"Error moving {image}: {e}")

pprint(images)
print(len(images))

# Dataset data in /dataset/ELPephants/data.csv

