import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from PIL import Image


CONTOUR_DIR = "/Users/kayoko/Documents/GitHub/elephant-identification/dataset/curvrank_ears_output/"
IMAGE_DIR = "/Users/kayoko/Documents/GitHub/elephant-identification/dataset/curvrank_ears/"
OUTPUT_DIR = "/Users/kayoko/Documents/GitHub/elephant-identification/dataset/anchor_aug_cropped"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "_annotations.coco.json")


def extract_bbox_from_contour(contour: list[list[int]]) -> list[float]:
    """Compute COCO-format bounding box [x_min, y_min, width, height] from contour points."""
    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]
    x_min = min(xs)
    y_min = min(ys)
    width = max(xs) - x_min
    height = max(ys) - y_min
    return [float(x_min), float(y_min), float(width), float(height)]


def extract_keypoints_from_contour(contour: list[list[int]]) -> list[float]:
    """Extract first and last points of contour as keypoints in COCO format [x1, y1, v1, x2, y2, v2]."""
    first_point = contour[0]
    last_point = contour[-1]
    return [
        float(first_point[0]), float(first_point[1]), 2,
        float(last_point[0]), float(last_point[1]), 2
    ]


def process_contour_file(contour_path: Path, image_id: int) -> tuple[dict, dict] | None:
    """
    Process a contour JSON file and return COCO format image info and annotation.
    
    Returns:
        Tuple of (image_info_dict, annotation_dict) or None if image not found
    """
    with open(contour_path) as f:
        contour_data = json.load(f)
    
    if not contour_data.get("contours") or len(contour_data["contours"]) == 0:
        raise ValueError(f"No contours found in {contour_path.name}")
    
    contour = contour_data["contours"][0]
    if len(contour) < 2:
        raise ValueError(f"Contour has fewer than 2 points in {contour_path.name}")
    
    image_filename = contour_path.stem + ".jpg"
    image_path = Path(IMAGE_DIR) / image_filename
    
    if not image_path.exists():
        image_filename_upper = contour_path.stem + ".JPG"
        image_path = Path(IMAGE_DIR) / image_filename_upper
        if not image_path.exists():
            raise ValueError(f"Image not found for {contour_path.name}")
        image_filename = image_filename_upper
    
    with Image.open(image_path) as img:
        width, height = img.size
    
    image_info = {
        "id": image_id,
        "license": 1,
        "file_name": image_filename,
        "height": height,
        "width": width,
        "date_captured": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "extra": {
            "name": image_filename
        }
    }
    
    bbox = extract_bbox_from_contour(contour)
    keypoints = extract_keypoints_from_contour(contour)
    area = bbox[2] * bbox[3]
    
    annotation = {
        "id": None,
        "image_id": image_id,
        "category_id": 1,
        "bbox": bbox,
        "area": area,
        "segmentation": [],
        "iscrowd": 0,
        "keypoints": keypoints
    }
    
    return image_info, annotation


def generate_coco_annotations() -> None:
    """Generate COCO format annotations from curvrank contour files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    coco_data = {
        "info": {
            "year": "2025",
            "version": "1",
            "description": "Generated from curvrank contour data",
            "contributor": "",
            "url": "",
            "date_created": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S+00:00")
        },
        "licenses": [
            {
                "id": 1,
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "name": "CC BY 4.0"
            }
        ],
        "categories": [
            {
                "id": 0,
                "name": "anchor",
                "supercategory": "none"
            },
            {
                "id": 1,
                "name": "elephant",
                "supercategory": "anchor",
                "keypoints": [
                    "new-point-2",
                    "new-point-3"
                ],
                "skeleton": [
                    [1, 2]
                ]
            }
        ],
        "images": [],
        "annotations": []
    }
    
    contour_dir_path = Path(CONTOUR_DIR)
    contour_files = sorted([f for f in contour_dir_path.iterdir() if f.suffix == ".json"])
    
    print(f"Found {len(contour_files)} contour files in {CONTOUR_DIR}")
    
    image_id = 0
    annotation_id = 1
    skipped_files = []
    
    for contour_path in contour_files:
        try:
            image_info, annotation = process_contour_file(contour_path, image_id)
            
            coco_data["images"].append(image_info)
            
            annotation["id"] = annotation_id
            coco_data["annotations"].append(annotation)
            annotation_id += 1
            
            # Copy image to output directory
            image_path = Path(IMAGE_DIR) / image_info["file_name"]
            destination = os.path.join(OUTPUT_DIR, image_info["file_name"])
            shutil.copy2(str(image_path), destination)
            
            image_id += 1
            print(f"✓ Processed: {contour_path.name}")
            
        except ValueError as e:
            skipped_files.append(contour_path.name)
            print(f"✗ ERROR: {e}")
        except Exception as e:
            skipped_files.append(contour_path.name)
            print(f"✗ ERROR processing {contour_path.name}: {e}")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(coco_data, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total images processed: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Skipped files: {len(skipped_files)}")
    if skipped_files:
        print(f"Skipped: {', '.join(skipped_files[:10])}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  - Images copied: {len(coco_data['images'])}")
    print(f"  - Annotations file: _annotations.coco.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    generate_coco_annotations()

