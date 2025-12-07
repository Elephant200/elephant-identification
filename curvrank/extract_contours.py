from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, Metadata

ROOT_DIR = "/Users/kayoko/Documents/GitHub/elephant-identification"

register_coco_instances(
    name="elephant-identification-train",
    metadata={},
    json_file=f"{ROOT_DIR}/dataset/curvrank/train/_annotations.coco.json",
    image_root=f"{ROOT_DIR}/dataset/curvrank/train",
)
register_coco_instances(
    name="elephant-identification-valid",
    metadata={},
    json_file=f"{ROOT_DIR}/dataset/curvrank/valid/_annotations.coco.json",
    image_root=f"{ROOT_DIR}/dataset/curvrank/valid",
)
register_coco_instances(
    name="elephant-identification-test",
    metadata={},
    json_file=f"{ROOT_DIR}/dataset/curvrank/test/_annotations.coco.json",
    image_root=f"{ROOT_DIR}/dataset/curvrank/test",
)