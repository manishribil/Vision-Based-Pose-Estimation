
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
PROJECT = Path(__file__).resolve().parent

INPUT_CSV = (
    PROJECT
    / "runs"
    / "phase8_formation_statistics"
    / "formation_statistics.csv"
)

OUTPUT_DIR = PROJECT / "runs" / "phase9_visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Phase 8 statistics
df = pd.read_csv(INPUT_CSV)

# Keep images with at least one drone pair
df = df[df["pairwise_distance_count"] > 0].copy()

if df.empty:
    raise ValueError("No images with pairwise distances were found.")

# Sort images by average normalized distance
df = df.sort_values("average_distance").reset_index(drop=True)

# Chart 1: Average distance across images
plt.figure(figsize=(14, 6))

plt.plot(
    range(1, len(df) + 1),
    df["average_distance"],
    marker=".",
    linestyle="-",
)

plt.xlabel("Images (sorted by average distance)")
plt.ylabel("Average normalized pairwise distance")
plt.title("Drone Formation Spacing Across Test Images")
plt.grid(True, alpha=0.3)
plt.tight_layout()

chart1 = OUTPUT_DIR / "average_distance_by_image.png"
plt.savefig(chart1, dpi=200)
plt.close()

# Chart 2: Distribution of average distances
plt.figure(figsize=(9, 6))

plt.hist(
    df["average_distance"],
    bins=15,
    edgecolor="black",
)

plt.xlabel("Average normalized pairwise distance")
plt.ylabel("Number of images")
plt.title("Distribution of Drone Formation Spacing")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

chart2 = OUTPUT_DIR / "distance_distribution.png"
plt.savefig(chart2, dpi=200)
plt.close()

# Save descriptive statistics
summary = df["average_distance"].describe()

summary_path = OUTPUT_DIR / "distance_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:
    file.write("FORMATION DISTANCE SUMMARY\n")
    file.write("==========================\n\n")
    file.write(f"Images analysed: {len(df)}\n\n")
    file.write(summary.to_string())
    file.write("\n")

print("\n========== PHASE 9 SUMMARY ==========")
print(f"Images analysed: {len(df)}")
print(f"Chart 1: {chart1}")
print(f"Chart 2: {chart2}")
print(f"Summary: {summary_path}")
print("=====================================")
