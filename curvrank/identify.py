import os
import numpy as np
import cv2
from tqdm import tqdm
from curvrank.contour import get_contour
from curvrank.curve import curvature, curvature_descriptors
from typing import Literal

from curvrank.preprocess import preprocess_images
from utils import get_all_images

TARGET_SIZE = 432
PADDING = 0.10 # 10% padding around the initial contour

def clamp(x, lower, upper):
    """Equivalent to max(lower, min(x, upper))"""
    return max(lower, min(x, upper))

def pipeline(image_paths: list[str]) -> list[tuple[str, list[np.ndarray]]]:
    """
    Pipeline to perform contour extraction, descriptor calculation, and LNBNN matching.
    """
    image_paths, views, names = preprocess_images(image_paths, output_dir="curvrank/preprocessed")
    print(f"Preprocessed {len(image_paths)} images")
    print(f"Views: {len(views)}")
    print(f"Names: {len(names)}")
    for ip, view, name in zip(image_paths, views, names):
        # Check files exist and view / name match
        if not os.path.exists(ip):
            print(f"File {ip} does not exist")
            continue
        if view != ip.split("_")[-1].split(".")[0]:
            print(f"View {view} does not match {ip.split('_')[-1]}")
            continue
        if name != "_".join(ip.split("/")[-1].split("_")[:-1]):
            print(f"Name {name} does not match {'_'.join(ip.split('/')[-1].split('_')[:-1])}")
            continue
    print(f"All images match")

if __name__ == "__main__":
    image_paths = get_all_images("processing/ELPephants/unannotated/certain")
    #preprocess_images(image_paths, output_dir="curvrank/preprocessed")
    pipeline(image_paths)

    # for name, descriptors in pipeline(image_paths, views, names):
    #     print(name, descriptors)