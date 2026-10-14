
import csv
import math
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent

INPUT_CSV = (
    PROJECT / "runs" / "phase6_formation_map"
    / "normalised_positions.csv"
)

OUTPUT_DIR = PROJECT / "runs" / "phase7_distance_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Group drone positions by image
drones_by_image = defaultdict(list)

with open(INPUT_CSV, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        drones_by_image[row["image"]].append(row)

distance_rows = []

for image_name, drones in drones_by_image.items():
    for i in range(len(drones)):
        for j in range(i + 1, len(drones)):
            drone_a = drones[i]
            drone_b = drones[j]

            dx = (
                float(drone_b["x_normalized"])
                - float(drone_a["x_normalized"])
            )
            dy = (
                float(drone_b["y_normalized"])
                - float(drone_a["y_normalized"])
            )

            distance = math.sqrt(dx**2 + dy**2)

            distance_rows.append({
                "image": image_name,
                "drone_a": drone_a["drone_id"],
                "drone_b": drone_b["drone_id"],
                "delta_x_normalized": dx,
                "delta_y_normalized": dy,
                "normalized_distance": distance,
            })

OUTPUT_CSV = OUTPUT_DIR / "normalized_pairwise_distances.csv"

fieldnames = [
    "image",
    "drone_a",
    "drone_b",
    "delta_x_normalized",
    "delta_y_normalized",
    "normalized_distance",
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(distance_rows)

print("\n========== PHASE 7 SUMMARY ==========")
print(f"Images analysed: {len(drones_by_image)}")
print(f"Drone pairs measured: {len(distance_rows)}")
print(f"Output CSV: {OUTPUT_CSV}")
print("=====================================")
