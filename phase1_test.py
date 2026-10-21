
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent

# Check for trained model first, fallback to base weights
model_path = PROJECT_DIR / "runs" / "detect" / "runs" / "drone" / "formation_v1" / "weights" / "best.pt"
if not model_path.exists():
    model_path = PROJECT_DIR / "yolo11n.pt"

# Check for test image location
image_path = PROJECT_DIR / "drone.jpg"
if not image_path.exists():
    image_path = PROJECT_DIR / "test_images" / "drone.jpg"

print(f"Loading YOLO model from: {model_path}...")
model = YOLO(str(model_path))
print("Model loaded successfully!")

print(f"Running inference on: {image_path}...")
results = model.predict(
    source=str(image_path),
    save=True,
    conf=0.25
)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            f"Detected: {model.names[class_id]} "
            f"| Confidence: {confidence:.2f}"
        )

print("Phase 1 YOLO test completed!")

