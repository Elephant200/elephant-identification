import inference
import supervision as sv
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
if api_key is None:
    raise ValueError("ROBOFLOW_API_KEY not found in .env file")

def get_prediction(model_id: str, image_path: str) -> sv.Detections:
    model = inference.get_model(f"elephant-identification-research/{model_id}", api_key=api_key)
    results = model.infer(image_path)[0]
    detections = sv.Detections.from_inference(results)
    return detections

if __name__ == "__main__":
    import sys
    
    model_id = sys.argv[1]
    image_path = sys.argv[2]

    print(get_prediction(model_id, image_path))