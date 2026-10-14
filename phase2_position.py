
from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "runs"
    / "drone"
    / "formation_v1"
    / "weights"
    / "best.pt"
)

TEST_DIR = (
    PROJECT_DIR
    / "Multiple Drones.v1i.yolov11"
    / "test"
    / "images"
)

OUTPUT_DIR = PROJECT_DIR / "runs" / "phase2_position"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
CSV_PATH = OUTPUT_DIR / "drone_positions.csv"

# 2. Check required files
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test images folder not found: {TEST_DIR}"
    )

IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 3. Load your trained YOLO model
model = YOLO(str(MODEL_PATH))

# 4. Detect drones and calculate their center points
image_files = [
    p for p in TEST_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
]

if not image_files:
    raise FileNotFoundError(
        f"No images found in: {TEST_DIR}"
    )

csv_rows = []

for image_path in sorted(image_files):
    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        device=0,
        verbose=False
    )

    result = results[0]
    image = result.orig_img.copy()

    if result.boxes is not None:
        for drone_id, box in enumerate(result.boxes, start=1):
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            confidence = float(box.conf[0].cpu().item())

            # Bounding-box center in image pixels
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Draw bounding box and center point
            cv2.rectangle(
                image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 0, 0),
                2
            )

            cv2.circle(
                image,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            label = f"D{drone_id}: ({center_x}, {center_y})"

            cv2.putText(
                image,
                label,
                (int(x1), max(int(y1) - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

            csv_rows.append({
                "image": image_path.name,
                "drone_id": drone_id,
                "confidence": round(confidence, 4),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "center_x": center_x,
                "center_y": center_y
            })

    # Save the annotated image
    output_path = IMAGE_OUTPUT_DIR / image_path.name

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not save image: {output_path}")

    print(
        f"{image_path.name}: "
        f"{0 if result.boxes is None else len(result.boxes)} "
        "drones detected"
    )

# 5. Save all coordinates in a CSV file
fieldnames = [
    "image", "drone_id", "confidence",
    "x1", "y1", "x2", "y2",
    "center_x", "center_y"
]

with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

print("\nPhase 2 completed!")
print(f"Annotated images: {IMAGE_OUTPUT_DIR}")
print(f"Position data: {CSV_PATH}")
print(f"Total images processed: {len(image_files)}")
print(f"Total drone detections: {len(csv_rows)}")
