
from pathlib import Path
import csv
from pathlib import Path
import torch
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
    csv_path = output_dir / "evaluation_metrics.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Determine execution device
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if not data_path.exists():
        if csv_path.exists():
            print(f"Dataset config not found at: {data_path}")
            print("Loading previously evaluated test-set metrics from evaluation_metrics.csv:\n")
            print("========== TEST RESULTS ==========")
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    print(f"{row[0]}: {row[1]} ({row[2]})")
            print(f"\nMetrics file: {csv_path}")
            print("(To rerun evaluation against the full test split, place Multiple Drones.v1i.yolov11/ in the project directory.)")
            return
        else:
            raise FileNotFoundError(f"Dataset config not found: {data_path}")

    print("Loading trained YOLO model...")
    model = YOLO(str(model_path))

    print(f"Evaluating on the test split using device: {device}...")

    results = model.val(
        data=str(data_path),
        split="test",
        device=device,
        workers=0,
    )

    metrics = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
    }

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
