
from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("Model loaded successfully!")

results = model.predict(
    source="drone.jpg",
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
