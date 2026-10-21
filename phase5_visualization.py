
import csv
import math
from pathlib import Path
from collections import defaultdict

import cv2

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# --------------------------------------------------
# 1. PROJECT PATHS & ARGUMENTS
# --------------------------------------------------

PROJECT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Phase 5: Drone Detection and Formation Visualization")
parser.add_argument("--images", type=str, default=None, help="Path to input images directory")
parser.add_argument("--limit", type=int, default=15, help="Number of visualizations to generate (-1 for all)")
args = parser.parse_args()

CSV_PATH = PROJECT / "runs" / "phase2_position" / "drone_positions.csv"

if args.images:
    IMAGE_DIR = Path(args.images)
else:
    dataset_dir = PROJECT / "Multiple Drones.v1i.yolov11" / "test" / "images"
    if dataset_dir.exists():
        IMAGE_DIR = dataset_dir
    else:
        IMAGE_DIR = PROJECT / "test_images"

OUTPUT_DIR = PROJECT / "runs" / "phase5_visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. CHECK INPUTS
# --------------------------------------------------

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

# --------------------------------------------------
# 3. READ CSV AND GROUP BY IMAGE
# --------------------------------------------------

detections_by_image = defaultdict(list)

with open(CSV_PATH, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        detections_by_image[row["image"]].append(row)


# --------------------------------------------------
# 4. PROCESS EACH IMAGE
# --------------------------------------------------

processed = 0
missing = 0

for image_name, drones in detections_by_image.items():
    if args.limit != -1 and processed >= args.limit:
        break

    image_path = IMAGE_DIR / image_name

    if image_path.exists():
        image = cv2.imread(str(image_path))
    else:
        image = None

    if image is None:
        # Create a synthetic 640x640 arena canvas if raw frame is not on disk
        image = np.full((640, 640, 3), 30, dtype=np.uint8)
        # Draw light grid lines
        for g in range(80, 640, 80):
            cv2.line(image, (g, 0), (g, 640), (45, 45, 45), 1)
            cv2.line(image, (0, g), (640, g), (45, 45, 45), 1)
        cv2.putText(
            image,
            f"Formation Arena: {image_name[:32]}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1
        )


    centres = []

    # Draw bounding boxes, centres and confidence labels
    for drone in drones:
        x1 = int(float(drone["x1"]))
        y1 = int(float(drone["y1"]))
        x2 = int(float(drone["x2"]))
        y2 = int(float(drone["y2"]))

        cx = int(float(drone["center_x"]))
        cy = int(float(drone["center_y"]))

        drone_id = drone["drone_id"]
        confidence = float(drone["confidence"])

        # Green bounding box
        cv2.rectangle(
            image, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

        # Red centre point
        cv2.circle(
            image, (cx, cy), 5, (0, 0, 255), -1
        )

        # Drone label
        label = f"Drone {drone_id} ({confidence:.2f})"

        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        centres.append((drone_id, cx, cy))

    # --------------------------------------------------
    # 5. DRAW PAIRWISE LINES AND STAGGERED LABELS
    # --------------------------------------------------

    pair_number = 0
    total_pairs = len(centres) * (len(centres) - 1) // 2

    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):

            id_a, ax, ay = centres[i]
            id_b, bx, by = centres[j]

            dx = bx - ax
            dy = by - ay

            distance = math.hypot(dx, dy)

            # Blue line connecting the drone centres
            cv2.line(
                image,
                (ax, ay),
                (bx, by),
                (255, 0, 0),
                2,
            )

            # Midpoint of the connecting line
            mx = (ax + bx) / 2
            my = (ay + by) / 2

            # Perpendicular direction to the line
            if distance > 0:
                perpendicular_x = -dy / distance
                perpendicular_y = dx / distance
            else:
                perpendicular_x = 0
                perpendicular_y = -1

            # Spread labels along the perpendicular direction
            offset = (pair_number - (total_pairs - 1) / 2) * 15

            label_x = int(mx + perpendicular_x * offset)
            label_y = int(my + perpendicular_y * offset)

            label = f"{distance:.0f} px"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            # Keep label coordinates within the image
            label_x = max(
                text_w // 2 + 3,
                min(label_x, image.shape[1] - text_w // 2 - 3),
            )
            label_y = max(
                text_h + 4,
                min(label_y, image.shape[0] - baseline - 4),
            )

            # Dark filled rectangle behind each distance label
            top_left = (
                label_x - text_w // 2 - 4,
                label_y - text_h - 4,
            )
            bottom_right = (
                label_x + text_w // 2 + 4,
                label_y + baseline + 3,
            )

            cv2.rectangle(
                image, top_left, bottom_right, (40, 40, 40), -1
            )

            # Yellow distance text
            cv2.putText(
                image,
                label,
                (label_x - text_w // 2, label_y),
                font,
                font_scale,
                (0, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

            pair_number += 1

    # --------------------------------------------------
    # 6. SAVE IMAGE
    # --------------------------------------------------

    output_path = OUTPUT_DIR / image_name

    if cv2.imwrite(str(output_path), image):
        processed += 1
    else:
        print(f"Could not save: {output_path}")


# --------------------------------------------------
# 7. SUMMARY
# --------------------------------------------------

print("\n========== PHASE 5 SUMMARY ==========")
print(f"Images with detections: {len(detections_by_image)}")
print(f"Images successfully saved: {processed}")
print(f"Images missing or unreadable: {missing}")
print(f"Output folder: {OUTPUT_DIR}")
print("=====================================")
