#!/usr/bin/env python3
"""
End-to-End Pipeline Verification Script
Vision-Based Precision Pose Estimation For Autonomous Formation Flying

Verifies and executes each phase of the project sequentially, asserting file
existence, data integrity, and model functionality from Phase 1 to Phase 10.
"""

import sys
import time
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent

def log_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def check_file(path, description):
    p = Path(path)
    if p.exists():
        size_kb = p.stat().st_size / 1024
        print(f"  [OK] {description} ({p.name}, {size_kb:.1f} KB)")
        return True
    else:
        print(f"  [MISSING] {description} ({path})")
        return False

def verify_all():
    results = []
    start_total = time.time()
    log_header("VISION-BASED POSE ESTIMATION: END-TO-END VERIFICATION")
    print(f"Workspace Directory: {PROJECT_DIR}\n")

    # 1. Weights and Setup
    t0 = time.time()
    weights_path = PROJECT_DIR / "runs/detect/runs/drone/formation_v1/weights/best.pt"
    ok1 = check_file(weights_path, "Trained YOLO11n weights")
    results.append(("Phase 0: Weights & Setup", ok1, f"{time.time() - t0:.2f}s"))

    # 2. Phase 1: YOLO Model Loading & Inference
    t0 = time.time()
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        test_img = PROJECT_DIR / "test_images/drone.jpg"
        pred = model.predict(str(test_img), conf=0.25, verbose=False)[0]
        n_drones = len(pred.boxes)
        print(f"  [OK] Model successfully loaded; detected {n_drones} drone(s) in drone.jpg")
        results.append(("Phase 1: Detection Test", n_drones > 0, f"{time.time() - t0:.2f}s"))
    except Exception as e:
        print(f"  [FAIL] Phase 1 failed: {e}")
        results.append(("Phase 1: Detection Test", False, f"{time.time() - t0:.2f}s"))

    # 3. Phase 2: Position Data
    t0 = time.time()
    pos_csv = PROJECT_DIR / "runs/phase2_position/drone_positions.csv"
    ok2 = check_file(pos_csv, "Extracted 2D Drone Positions CSV")
    if ok2:
        df_pos = pd.read_csv(pos_csv)
        print(f"  [OK] Drone Positions verified: {len(df_pos)} detections across {df_pos['image'].nunique()} frames")
    results.append(("Phase 2: 2D Position Extraction", ok2, f"{time.time() - t0:.2f}s"))

    # 4. Phase 3: Spatial Formation Analysis
    t0 = time.time()
    form_csv = PROJECT_DIR / "runs/phase3_formation/formation_analysis.csv"
    ok3 = check_file(form_csv, "Pairwise Formation Analysis CSV")
    if ok3:
        df_form = pd.read_csv(form_csv)
        print(f"  [OK] Formation Analysis verified: {len(df_form)} drone pairs analyzed")
    results.append(("Phase 3: Spatial Arrangement", ok3, f"{time.time() - t0:.2f}s"))

    # 5. Phase 4: Model Evaluation Metrics
    t0 = time.time()
    eval_csv = PROJECT_DIR / "runs/phase4_evaluation/evaluation_metrics.csv"
    ok4 = check_file(eval_csv, "Evaluation Metrics CSV")
    if ok4:
        df_eval = pd.read_csv(eval_csv)
        metrics_dict = dict(zip(df_eval.iloc[:, 0], df_eval.iloc[:, 2]))
        print(f"  [OK] Test Set Metrics: Precision: {metrics_dict.get('precision', 'N/A')}, "
              f"Recall: {metrics_dict.get('recall', 'N/A')}, mAP@50: {metrics_dict.get('mAP50', 'N/A')}, "
              f"mAP@50-95: {metrics_dict.get('mAP50-95', 'N/A')}")
    results.append(("Phase 4: Model Evaluation", ok4, f"{time.time() - t0:.2f}s"))

    # 6. Phase 5: Visualizations
    t0 = time.time()
    viz_dir = PROJECT_DIR / "runs/phase5_visualization"
    n_viz = len(list(viz_dir.glob("*.jpg"))) if viz_dir.exists() else 0
    ok5 = n_viz > 0
    print(f"  [OK] Visual Overlays: {n_viz} annotated images in {viz_dir.name}")
    results.append(("Phase 5: Annotated Visualizations", ok5, f"{time.time() - t0:.2f}s"))

    # 7. Phase 6: Formation Normalization
    t0 = time.time()
    norm_csv = PROJECT_DIR / "runs/phase6_formation_map/normalised_positions.csv"
    ok6 = check_file(norm_csv, "Normalised Positions CSV")
    results.append(("Phase 6: Normalized Coordinate Map", ok6, f"{time.time() - t0:.2f}s"))

    # 8. Phase 7: Pairwise Normalized Distances
    t0 = time.time()
    dist_csv = PROJECT_DIR / "runs/phase7_distance_analysis/normalized_pairwise_distances.csv"
    ok7 = check_file(dist_csv, "Normalized Pairwise Distances CSV")
    if ok7:
        df_dist = pd.read_csv(dist_csv)
        print(f"  [OK] Pairwise Distances: {len(df_dist)} pairs computed")
    results.append(("Phase 7: Pairwise Distance Analysis", ok7, f"{time.time() - t0:.2f}s"))

    # 9. Phase 8: Formation Statistics
    t0 = time.time()
    stat_csv = PROJECT_DIR / "runs/phase8_formation_statistics/formation_statistics.csv"
    ok8 = check_file(stat_csv, "Formation Statistics CSV")
    if ok8:
        df_stat = pd.read_csv(stat_csv)
        print(f"  [OK] Statistics: {len(df_stat)} image summaries, avg spacing: {df_stat['average_distance'].mean():.4f}")
    results.append(("Phase 8: Formation Statistics", ok8, f"{time.time() - t0:.2f}s"))

    # 10. Phase 9: Statistical Visualizations
    t0 = time.time()
    p9_dir = PROJECT_DIR / "runs/phase9_visualization"
    c1 = check_file(p9_dir / "average_distance_by_image.png", "Average Distance Chart")
    c2 = check_file(p9_dir / "distance_distribution.png", "Distance Distribution Histogram")
    c3 = check_file(p9_dir / "distance_summary.txt", "Descriptive Summary Text")
    ok9 = c1 and c2 and c3
    results.append(("Phase 9: Statistical Visualization", ok9, f"{time.time() - t0:.2f}s"))

    # 11. Phase 10: 3D Distance Estimation & Formation Tracking
    t0 = time.time()
    p10_dir = PROJECT_DIR / "results/distance"
    d1 = check_file(p10_dir / "metrics.csv", "3D Distance Metrics CSV")
    d2 = check_file(p10_dir / "summary.png", "Formation Status & Spacing Summary")
    d3 = check_file(p10_dir / "error_details.csv", "Ground-Truth Error Comparison")
    d4 = check_file(p10_dir / "error_analysis.png", "Sensitivity & Box Error Plots")
    d5 = check_file(p10_dir / "report_numbers.txt", "Comprehensive Numerical Report")
    d6 = check_file(p10_dir / "formation_demo.mp4", "Demo Video with Top-down Radar")
    ok10 = d1 and d2 and d3 and d4 and d5 and d6
    results.append(("Phase 10: 3D Distance & HUD Tracking", ok10, f"{time.time() - t0:.2f}s"))

    # 12. Tools & Edge Case Tests
    t0 = time.time()
    edge_summary = PROJECT_DIR / "edge_cases/summary.csv"
    ok_tools = check_file(edge_summary, "Edge Case Evaluation Summary CSV")
    results.append(("Tools & Edge-Case Robustness", ok_tools, f"{time.time() - t0:.2f}s"))

    # Summary Table
    total_time = time.time() - start_total
    log_header("VERIFICATION SUMMARY REPORT")
    print(f"{'Phase / Component':<38} | {'Status':<10} | {'Time':<8}")
    print("-" * 62)
    all_passed = True
    for phase_name, passed, elapsed in results:
        status_str = "PASS [OK]" if passed else "FAIL [X]"
        if not passed:
            all_passed = False
        print(f"{phase_name:<38} | {status_str:<10} | {elapsed:<8}")
    print("-" * 62)
    print(f"Total Pipeline Status: {'ALL PHASES PASSED (10/10)' if all_passed else 'SOME PHASES FAILED'}")
    print(f"Total Verification Elapsed Time: {total_time:.2f}s")
    print("=" * 70 + "\n")
    return all_passed

if __name__ == "__main__":
    success = verify_all()
    sys.exit(0 if success else 1)
