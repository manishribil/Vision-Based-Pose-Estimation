
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt

# Project paths
PROJECT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Phase 6: Formation Mapping")
parser.add_argument("--images", type=str, default=None, help="Path to images directory")
parser.add_argument("--limit-plots", type=int, default=10, help="Number of formation map plots to generate (default: 10, -1 for all)")
args = parser.parse_args()

CSV_PATH = PROJECT / "runs" / "phase2_position" / "drone_positions.csv"
if args.images:
    IMAGE_DIR = Path(args.images)
else:
    IMAGE_DIR = PROJECT / "Multiple Drones.v1i.yolov11" / "test" / "images"

OUTPUT_DIR = PROJECT / "runs" / "phase6_formation_map"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

# Group detections by image
detections_by_image = defaultdict(list)

with open(CSV_PATH, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        detections_by_image[row["image"]].append(row)

summary_rows = []
saved_plots = 0

for image_name, drones in detections_by_image.items():
    image_path = IMAGE_DIR / image_name
    image = cv2.imread(str(image_path)) if image_path.exists() else None

    if image is not None:
        height, width = image.shape[:2]
    else:
        # Standard resolution for the dataset video frames
        height, width = 640, 640

    points = []

    for drone in drones:
        x = float(drone["center_x"])
        y = float(drone["center_y"])

        # Normalise image coordinates to the range 0–1
        x_norm = x / width
        y_norm = y / height

        points.append((x_norm, y_norm))

        summary_rows.append({
            "image": image_name,
            "drone_id": drone["drone_id"],
            "center_x_px": x,
            "center_y_px": y,
            "image_width": width,
            "image_height": height,
            "x_normalized": x_norm,
            "y_normalized": y_norm,
        })

    # Plot the formation in normalised image coordinates (up to limit_plots or all if -1)
    if args.limit_plots == -1 or saved_plots < args.limit_plots:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(xs, ys, s=100, marker="o", color="tab:blue")

        for index, (x, y) in enumerate(points):
            ax.annotate(
                f"D{index + 1}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontweight="bold"
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # Image y-coordinates increase downward
        ax.set_xlabel("Normalised X position")
        ax.set_ylabel("Normalised Y position")
        ax.set_title(f"Drone Formation\n{image_name[:45]}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = OUTPUT_DIR / f"{Path(image_name).stem}_map.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        saved_plots += 1


# Save all normalised coordinates to CSV
csv_output = OUTPUT_DIR / "normalised_positions.csv"

with open(csv_output, "w", newline="", encoding="utf-8") as file:
    fieldnames = [
        "image",
        "drone_id",
        "center_x_px",
        "center_y_px",
        "image_width",
        "image_height",
        "x_normalized",
        "y_normalized",
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

print("\n========== PHASE 6 SUMMARY ==========")
print(f"Images mapped: {saved_plots}")
print(f"Drone positions recorded: {len(summary_rows)}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Coordinate CSV: {csv_output}")
print("=====================================")
