
import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

INPUT_CSV = (
    PROJECT / "runs" / "phase7_distance_analysis"
    / "normalized_pairwise_distances.csv"
)

POSITIONS_CSV = (
    PROJECT / "runs" / "phase6_formation_map"
    / "normalised_positions.csv"
)

OUTPUT_DIR = PROJECT / "runs" / "phase8_formation_statistics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

distances = pd.read_csv(INPUT_CSV)
positions = pd.read_csv(POSITIONS_CSV)

# Count detected drones in each image
drone_counts = (
    positions.groupby("image")["drone_id"]
    .count()
    .rename("detected_drone_count")
)

# Summarize pairwise distances for each image
if distances.empty:
    statistics = pd.DataFrame(
        columns=[
            "image",
            "average_distance",
            "minimum_distance",
            "maximum_distance",
            "distance_std_dev",
        ]
    )
else:
    statistics = (
        distances.groupby("image")["normalized_distance"]
        .agg(
            average_distance="mean",
            minimum_distance="min",
            maximum_distance="max",
            distance_std_dev="std",
        )
        .reset_index()
    )

# Include images with no pairwise distances
statistics = (
    statistics.set_index("image")
    .reindex(drone_counts.index)
    .reset_index()
)

statistics = statistics.merge(
    drone_counts.reset_index(),
    on="image",
    how="left",
)

statistics["pairwise_distance_count"] = (
    distances.groupby("image").size()
    .reindex(statistics["image"])
    .fillna(0)
    .astype(int)
    .to_numpy()
)

OUTPUT_CSV = OUTPUT_DIR / "formation_statistics.csv"
statistics.to_csv(OUTPUT_CSV, index=False)

print("\n========== PHASE 8 SUMMARY ==========")
print(f"Images summarized: {len(statistics)}")
print(f"Images with at least one pair: "
      f"{(statistics['pairwise_distance_count'] > 0).sum()}")
print(f"Output CSV: {OUTPUT_CSV}")
print("\nFirst five rows:")
print(statistics.head().to_string(index=False))
print("=====================================")
