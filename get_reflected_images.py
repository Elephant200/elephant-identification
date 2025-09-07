import os
import shutil
import cv2
from utils import is_image
from tqdm import tqdm

root_dir = "/Users/kayoko/Documents/GitHub/elephant-identification"

for image in tqdm(os.listdir(f"{root_dir}/dataset/ELPephants"), desc="Reflecting images"):
    if not is_image(f"{root_dir}/dataset/ELPephants/{image}"):
        continue
    try:
        source_path = f"{root_dir}/dataset/ELPephants/{image}"
        dest_path = f"{root_dir}/dataset/ELPephants/{image.split('.jpg')[0]}_reflected.jpg"
        # reflect the image
        image = cv2.imread(source_path)
        image = cv2.flip(image, 1)
        cv2.imwrite(dest_path, image)
    except Exception as e:
        print(f"Error reflecting {source_path}: {e}")