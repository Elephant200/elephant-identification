import inference
import supervision as sv
import cv2
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
if api_key is None:
    raise ValueError("ROBOFLOW_API_KEY not found in .env file")

def get_prediction(model_id: str, image_path: str) -> sv.Detections:
    model = inference.get_model(f"elephant-identification-research/{model_id}", api_key=api_key)
    image = cv2.imread(image_path)
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)
    return detections

if __name__ == "__main__":
    model_id = "20"
    image_path = "images/all_elephant_images/ahmed/ahmed_8.jpg"
    print(get_prediction(model_id, image_path))