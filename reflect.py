import os
import shutil
import cv2
from utils import is_image

root_dir = "/Users/kayoko/Documents/GitHub/elephant-identification"

for image in os.listdir(f"{root_dir}/images/ELPephants"):
    if not is_image(f"{root_dir}/images/ELPephants/{image}"):
        continue
    try:
        source_path = f"{root_dir}/images/ELPephants/{image}"
        dest_path_1 = f"{root_dir}/images/ELPephants-reflected/{image}"
        dest_path_2 = f"{root_dir}/images/ELPephants-reflected/{image.split('.jpg')[0]}_reflected.jpg"
        # reflect the image
        image = cv2.imread(source_path)
        image = cv2.flip(image, 1)
        cv2.imwrite(dest_path_2, image)
        shutil.copy2(source_path, dest_path_1)
        
    except Exception as e:
        print(f"Error reflecting {source_path}: {e}")