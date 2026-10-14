
from pathlib import Path
import csv
from ultralytics import YOLO


def main():
    project = Path(__file__).resolve().parent

    model_path = (
        project / "runs" / "detect" / "runs"
        / "drone" / "formation_v1" / "weights" / "best.pt"
    )

    data_path = (
        project / "Multiple Drones.v1i.yolov11" / "data.yaml"
    )

    output_dir = project / "runs" / "phase4_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    print("Loading trained YOLO model...")
    model = YOLO(str(model_path))

    print("Evaluating on the test split...")

    results = model.val(
        data=str(data_path),
        split="test",
        device=0,
        workers=0,
    )

    metrics = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
    }

    csv_path = output_dir / "evaluation_metrics.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Score", "Percentage"])

        for name, score in metrics.items():
            writer.writerow([
                name,
                f"{score:.4f}",
                f"{score * 100:.2f}%",
            ])

    print("\n========== TEST RESULTS ==========")

    for name, score in metrics.items():
        print(f"{name}: {score:.4f} ({score * 100:.2f}%)")

    print(f"\nMetrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
