"""
Uses SAM2 to segment the contours of elephants in the dataset used for CurvRank implementation.
"""

from copy import deepcopy
import json
from typing import Dict, List, Literal

import argparse
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from inference.models.sam2 import SegmentAnything2
from inference.core.entities.requests.sam2 import Sam2PromptSet, Sam2Prompt, Box
from inference.core.utils.postprocess import masks2poly

load_dotenv("../.env")

def load_coco_json(path: str) -> Dict:
    return json.load(open(path, "r"))

def mask_to_coco_polys(mask: np.ndarray) -> List[List[float]]:
    """Convert a boolean mask to COCO polygons.

    Args:
        mask: Boolean mask (H, W).

    Returns:
        List of flattened polygons.
    """
    polys = masks2poly(mask)
    out: List[List[float]] = []
    for p in polys:
        if p.shape[0] >= 3:
            flat = p.flatten().astype(float).tolist()
            if len(flat) >= 6:
                out.append(flat)
    return out

def coco_bbox_to_sam2_bbox(bbox: List[float]) -> Box:
    """
    Convert a COCO bounding box (top left x, top left y, width, height) to a SAM2 bounding box (center x, center y, width, height).

    Args:
        bbox: List of floats (x, y, width, height).

    Returns:
        Box: SAM2 bounding box.
    """
    return Box(x=bbox[0] + bbox[2]/2, y=bbox[1] + bbox[3]/2, width=bbox[2], height=bbox[3])

def get_elephants(coco_json: Dict, root_dir: str, mode: Literal["train", "valid", "test"]) -> List[Dict]:
    elephants = list()
    for image in coco_json["images"]:
        elephants.append({
            "image_id": image["id"],
            "image_path": f"{root_dir}/dataset/curvrank/{mode}/{image['file_name']}",
            "image_width": image["width"],
            "image_height": image["height"],
        })
    return elephants

if __name__ == "__main__":
    # set up argparse
    parser = argparse.ArgumentParser(
        description="Convert bounding box annotations to segmentation masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--force", action="store_true", help="Force re-segmentation of all images")
    args = parser.parse_args()

    mode = "valid" # Run this code once for each of train, valid, and test

    force: bool = args.force

    ROOT_DIR = "/Users/kayoko/Documents/GitHub/elephant-identification"
    upload_to_roboflow = True
    coco_json = load_coco_json(f"/Users/kayoko/Documents/GitHub/elephant-identification/dataset/curvrank/{mode}/_annotations.coco.json")
    elephants = get_elephants(coco_json, ROOT_DIR, mode)

    new_coco_json = deepcopy(coco_json)
    sam2 = SegmentAnything2(model_id="sam2/hiera_large")

    cur_elephant = elephants[0]
    cur_elephant_idx = 0
    annotations = coco_json["annotations"]
    changed = True
    try:
        for i, annotation in enumerate(tqdm(annotations, desc="Processing annotations")):
            if not force and annotation["segmentation"] != []: # Already segmented
                continue
            while not annotation["image_id"] == cur_elephant["image_id"]:
                cur_elephant_idx += 1
                cur_elephant = elephants[cur_elephant_idx]
                changed = True

            if changed:
                sam2.embed_image(cur_elephant["image_path"])
                changed = False
            
            base_image = cur_elephant["image_path"]
            coco_bbox = annotation["bbox"]

            prompts = Sam2PromptSet(prompts=[Sam2Prompt(box=coco_bbox_to_sam2_bbox(coco_bbox))])
            contours = sam2.segment_image(base_image, prompts=prompts)
            contours = mask_to_coco_polys(contours[0])
            
            new_coco_json["annotations"][i]["segmentation"] = contours

    except KeyboardInterrupt:
        print("Execution interrupted by user. Saving progress...")
    finally:
        print("Saving...")
        with open(f"{ROOT_DIR}/dataset/curvrank/{mode}/_annotations.coco.json", "w") as f:
            json.dump(new_coco_json, f, indent=4)
    

    # # Visualize conversion for one of the images
    # image_path = "/Users/kayoko/Downloads/African and Forest Elephants.v4i.yolov11/train/images/-calf-in-fenced-enclosure-with-hay-bales-at-walt-disney-world-animal-kingdom_jpg.rf.4c66565104ce0d6436f425755ad0b4cf.jpg"
    # boxes_xywh = [[209,6,94.4,204.8]]
    # sam2 = SegmentAnything2(model_id="sam2/hiera_large")
    # sam2.embed_image(image_path)
    # prompts = Sam2PromptSet(prompts=[Sam2Prompt(box=coco_bbox_to_sam2_bbox(boxes_xywh[0]))])
    # contours = sam2.segment_image(image_path, prompts=prompts)
    # contours = mask_to_coco_polys(contours[0])
    # contours = [np.array([[int(contour[i]), int(contour[i+1])] for i in range(0, len(contour), 2)], dtype=np.int32) for contour in contours]


    # # Draw contours on image
    # image = cv2.imread(image_path)
    # cv2.drawContours(image, contours, -1, (0, 0, 255), 2) 
    # cv2.imshow("Contours", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()