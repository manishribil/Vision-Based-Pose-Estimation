
from pathlib import Path
import csv
import math
from collections import defaultdict

# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent

INPUT_CSV = (
    PROJECT_DIR
    / "runs"
    / "phase2_position"
    / "drone_positions.csv"
)

OUTPUT_DIR = PROJECT_DIR / "runs" / "phase3_formation"
OUTPUT_CSV = OUTPUT_DIR / "formation_analysis.csv"

# 2. Check the input file exists
if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Phase 2 CSV not found: {INPUT_CSV}\n"
        "Run phase2_position.py first."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 3. Read detections and group them by image
detections_by_image = defaultdict(list)

with INPUT_CSV.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        detections_by_image[row["image"]].append({
            "drone_id": int(row["drone_id"]),
            "x": float(row["center_x"]),
            "y": float(row["center_y"]),
            "confidence": float(row["confidence"])
        })

# 4. Analyse relative positions and pairwise separation
analysis_rows = []

for image_name, drones in detections_by_image.items():
    # Sort by horizontal position for easier interpretation
    drones.sort(key=lambda drone: drone["x"])

    for drone in drones:
        drone["position_label"] = (
            "Left" if drone["x"] < 0 else "Right"
        )

    for i in range(len(drones)):
        for j in range(i + 1, len(drones)):
            drone_a = drones[i]
            drone_b = drones[j]

            dx = drone_b["x"] - drone_a["x"]
            dy = drone_b["y"] - drone_a["y"]

            pixel_distance = math.sqrt(dx**2 + dy**2)

            horizontal = (
                "B is right of A" if dx > 0
                else "B is left of A" if dx < 0
                else "B is horizontally aligned with A"
            )

            vertical = (
                "B is below A" if dy > 0
                else "B is above A" if dy < 0
                else "B is vertically aligned with A"
            )

            relation = f"{horizontal}; {vertical}"

            analysis_rows.append({
                "image": image_name,
                "drone_a_center": f"({drone_a['x']}, {drone_a['y']})",
                "drone_b_center": f"({drone_b['x']}, {drone_b['y']})",
                "delta_x_pixels": round(dx, 2),
                "delta_y_pixels": round(dy, 2),
                "pixel_separation": round(pixel_distance, 2),
                "relative_position": relation
            })
# 5. Save the analysis
fieldnames = [
    "image",
    "drone_a_center",
    "drone_b_center",
    "delta_x_pixels",
    "delta_y_pixels",
    "pixel_separation",
    "relative_position"
]

with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(analysis_rows)

# 6. Print summary
print("Phase 3 completed!")
print(f"Images analysed: {len(detections_by_image)}")
print(f"Drone pairs analysed: {len(analysis_rows)}")
print(f"Results saved to: {OUTPUT_CSV}")
