import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from PIL import Image
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
if api_key is None:
    raise ValueError("ROBOFLOW_API_KEY not found in .env file")

CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key=api_key
)
config = InferenceConfiguration(
    confidence_threshold=0.05,
    iou_threshold=0
)
CLIENT.configure(config)

MODEL_URL = "anchor-extraction-bwwlq/3"
IMAGE_DIR = "/Users/kayoko/Documents/GitHub/elephant-identification/processing/ELPephants/unannotated/probable/"
OUTPUT_DIR = "/Users/kayoko/Documents/GitHub/elephant-identification/dataset_creation/train/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "_annotations.coco.json")


def convert_bbox_to_coco(x_center: float, y_center: float, width: float, height: float) -> list[float]:
    """Convert center-based bbox to top-left corner-based bbox for COCO format."""
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    return [x_min, y_min, width, height]


def process_image(image_path: str, image_id: int) -> tuple[dict, list[dict]]:
    """
    Process a single image and return COCO format image info and annotations.
    
    Returns:
        Tuple of (image_info_dict, list_of_annotation_dicts)
    """
    # Get image dimensions
    with Image.open(image_path) as img:
        width, height = img.size
    
    # Get filename
    filename = os.path.basename(image_path)
    
    # Call the model
    result = CLIENT.infer(image_path, model_id=MODEL_URL)
    
    # Check if there are any predictions
    if not result.get("predictions"):
        raise ValueError(f"No detections found for {filename}")
    
    # Create image info
    image_info = {
        "id": image_id,
        "license": 1,
        "file_name": filename,
        "height": height,
        "width": width,
        "date_captured": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "extra": {
            "name": filename
        }
    }
    
    # Create annotations for each detection
    annotations = []
    for prediction in result["predictions"]:
        # Convert bbox from center format to top-left format
        bbox = convert_bbox_to_coco(
            prediction["x"],
            prediction["y"],
            prediction["width"],
            prediction["height"]
        )
        
        # Calculate area
        area = bbox[2] * bbox[3]
        
        # Extract keypoints in COCO format: [x1, y1, v1, x2, y2, v2, ...]
        # v=2 means visible (COCO visibility flag)
        keypoints = []
        if "keypoints" in prediction and len(prediction["keypoints"]) >= 2:
            # Assuming keypoints are ordered: new-point-2, new-point-3
            for kp in prediction["keypoints"]:
                keypoints.extend([kp["x"], kp["y"], 2])
        
        annotation = {
            "id": None,  # Will be assigned later
            "image_id": image_id,
            "category_id": 1,  # elephant category
            "bbox": bbox,
            "area": area,
            "segmentation": [],
            "iscrowd": 0,
            "keypoints": keypoints
        }
        annotations.append(annotation)
    
    return image_info, annotations


def generate_coco_annotations() -> None:
    """Generate COCO format annotations for all images in the directory."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize COCO structure
    coco_data = {
        "info": {
            "year": "2025",
            "version": "2",
            "description": "Exported from roboflow.com",
            "contributor": "",
            "url": "https://public.roboflow.com/object-detection/undefined",
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
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_dir_path = Path(IMAGE_DIR)
    image_files = sorted([
        f for f in image_dir_path.iterdir()
        if f.suffix in image_extensions
    ])
    
    print(f"Found {len(image_files)} images in {IMAGE_DIR}")
    
    # Process each image
    image_id = 0
    annotation_id = 1
    skipped_images = []
    
    for image_path in image_files:
        try:
            image_info, annotations = process_image(str(image_path), image_id)
            
            # Add image info
            coco_data["images"].append(image_info)
            
            # Add annotations with proper IDs
            for annotation in annotations:
                annotation["id"] = annotation_id
                coco_data["annotations"].append(annotation)
                annotation_id += 1
            
            # Copy image to output directory
            destination = os.path.join(OUTPUT_DIR, image_path.name)
            shutil.copy2(str(image_path), destination)
            
            image_id += 1
            print(f"✓ Processed: {image_path.name} ({len(annotations)} detections)")
            
        except ValueError as e:
            skipped_images.append(image_path.name)
            print(f"✗ ERROR: {e}")
        except Exception as e:
            skipped_images.append(image_path.name)
            print(f"✗ ERROR processing {image_path.name}: {e}")
    
    # Save to JSON file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(coco_data, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total images processed: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Skipped images: {len(skipped_images)}")
    if skipped_images:
        print(f"Skipped: {', '.join(skipped_images)}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  - Images copied: {len(coco_data['images'])}")
    print(f"  - Annotations file: _annotations.coco.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    generate_coco_annotations()
