#!/usr/bin/env python3
"""
Live Demonstration Script for Mini-Project Review & Evaluation
Vision-Based Precision Pose Estimation For Autonomous Formation Flying

Features:
- Real-time or single-frame drone detection using trained YOLO11n weights
- Pinhole camera 3D distance estimation (Z = f * W / w)
- Lateral & elevation spatial displacement (X, Y in meters)
- Multi-drone formation tracking: nearest-neighbour spacing in drone-widths
- Formation status classification: Maintained (Green), Too Close (Red), Outside Threshold (Orange)
- Real-time top-down bird's-eye radar inset
- Live telemetry Heads-Up Display (HUD)
"""

import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_DIR / "runs/detect/runs/drone/formation_v1/weights/best.pt"
DEFAULT_TEST_IMAGE = PROJECT_DIR / "test_images/drone.jpg"

REAL_DRONE_WIDTH_M = 0.46  # 3DR Solo diagonal motor-to-motor width
DEFAULT_HFOV_DEG = 60.0
IMG_SIZE = 640

def compute_focal_length(img_width, hfov_deg):
    return (img_width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)

def topdown_radar_map(drones_3d, size=200):
    """Generates an aerial bird's-eye view inset of drone positions relative to the camera."""
    radar = np.full((size, size, 3), 28, dtype=np.uint8)
    # Range rings
    cv2.circle(radar, (size // 2, size - 10), int(size * 0.4), (55, 55, 55), 1)
    cv2.circle(radar, (size // 2, size - 10), int(size * 0.75), (45, 45, 45), 1)
    # Camera marker
    cam_pt = (size // 2, size - 10)
    cv2.drawMarker(radar, cam_pt, (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 10, 1)

    if not drones_3d:
        cv2.putText(radar, "TOP VIEW", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        return radar

    z_max = max(max(d["Z_m"] for d in drones_3d) * 1.25, 2.0)
    x_max = max(max(abs(d["X_m"]) for d in drones_3d) * 1.4, 1.5)

    for d in drones_3d:
        px = int(size / 2 + (d["X_m"] / x_max) * (size / 2 - 20))
        py = int(size - 12 - (d["Z_m"] / z_max) * (size - 30))
        px = max(10, min(size - 10, px))
        py = max(10, min(size - 10, py))
        cv2.circle(radar, (px, py), 5, (0, 230, 255), -1)
        cv2.putText(radar, f"D{d['id']}", (px + 6, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    cv2.putText(radar, "TOP VIEW (XZ)", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    cv2.putText(radar, f"Range: {z_max:.1f}m", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
    return radar

def process_frame(frame, model, focal_px, drone_width_m=0.46, min_sep=3.0, max_sep=10.0, conf=0.25):
    h, w = frame.shape[:2]
    res = model.predict(frame, conf=conf, verbose=False)[0]

    detections = []
    if res.boxes is not None and len(res.boxes) > 0:
        for idx, box in enumerate(res.boxes, start=1):
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            c = float(box.conf[0].cpu().item())
            bw = max(x2 - x1, 1.0)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            z_m = (focal_px * drone_width_m) / bw
            x_m = (cx - w / 2.0) * z_m / focal_px
            y_m = (cy - h / 2.0) * z_m / focal_px

            detections.append({
                "id": idx, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "cy": cy, "bw": bw, "conf": c,
                "X_m": x_m, "Y_m": y_m, "Z_m": z_m
            })

    # Formation and nearest neighbour calculations
    nn_spacings = []
    if len(detections) >= 2:
        for i, d1 in enumerate(detections):
            min_dist = float("inf")
            min_widths = float("inf")
            partner_id = -1
            for j, d2 in enumerate(detections):
                if i == j:
                    continue
                pix_sep = math.hypot(d1["cx"] - d2["cx"], d1["cy"] - d2["cy"])
                avg_w = (d1["bw"] + d2["bw"]) / 2.0
                sep_widths = pix_sep / avg_w
                if sep_widths < min_widths:
                    min_widths = sep_widths
                    partner_id = d2["id"]
                    min_dist = math.sqrt((d1["X_m"] - d2["X_m"])**2 + (d1["Y_m"] - d2["Y_m"])**2 + (d1["Z_m"] - d2["Z_m"])**2)
            d1["nn_id"] = partner_id
            d1["nn_widths"] = min_widths
            d1["nn_dist_m"] = min_dist
            nn_spacings.append(min_widths)

    # Determine formation status
    if len(detections) == 0:
        status, status_col = "NO DRONES DETECTED", (0, 0, 255)
    elif len(detections) == 1:
        status, status_col = "SINGLE DRONE (STANDALONE)", (180, 180, 180)
    else:
        min_s = min(nn_spacings)
        max_s = max(nn_spacings)
        if min_s < min_sep:
            status, status_col = f"WARNING: TOO CLOSE (<{min_sep:.0f}w)", (0, 0, 255)
        elif max_s > max_sep:
            status, status_col = f"ALERT: OUTSIDE THRESHOLD (>{max_sep:.0f}w)", (0, 140, 255)
        else:
            status, status_col = "FORMATION MAINTAINED [NOMINAL]", (0, 220, 0)

    # Visualization
    annotated = frame.copy()

    # Draw inter-drone formation link lines
    if len(detections) >= 2:
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                p1 = (int(detections[i]["cx"]), int(detections[i]["cy"]))
                p2 = (int(detections[j]["cx"]), int(detections[j]["cy"]))
                cv2.line(annotated, p1, p2, (255, 200, 0), 1, cv2.LINE_AA)

    # Draw bounding boxes and labels
    for d in detections:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 230, 0), 2)
        cv2.circle(annotated, (int(d["cx"]), int(d["cy"])), 4, (0, 0, 255), -1)

        tag = f"D{d['id']} | Z={d['Z_m']:.2f}m ({d['conf']:.2f})"
        cv2.putText(annotated, tag, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        offset_tag = f"X:{d['X_m']:+.1f}m Y:{d['Y_m']:+.1f}m"
        cv2.putText(annotated, offset_tag, (x1, min(y2 + 16, h - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # Top-Left Telemetry HUD
    hud_h = 100 if len(detections) > 1 else 75
    cv2.rectangle(annotated, (0, 0), (450, hud_h), (15, 15, 15), -1)
    cv2.putText(annotated, f"FORMATION TRACKING: {len(detections)} DRONES", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(annotated, f"Status: {status}", (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_col, 2)

    if len(detections) >= 2:
        cv2.putText(annotated, f"Spacing: {min(nn_spacings):.1f} - {max(nn_spacings):.1f} drone-widths",
                    (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        z_min = min(d["Z_m"] for d in detections)
        z_max = max(d["Z_m"] for d in detections)
        cv2.putText(annotated, f"Depth Extent: {z_min:.1f}m to {z_max:.1f}m (W={drone_width_m}m)",
                    (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # Bottom-Right Radar Inset
    radar = topdown_radar_map(detections, size=180)
    rh, rw = radar.shape[:2]
    annotated[h - rh - 8:h - 8, w - rw - 8:w - 8] = radar

    return annotated, detections, status

def main():
    parser = argparse.ArgumentParser(description="Live Demonstration of Vision-Based Formation Pose Estimation")
    parser.add_argument("--source", default=str(DEFAULT_TEST_IMAGE), help="Input image path, video path, or '0' for live webcam")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to trained YOLO model weights")
    parser.add_argument("--hfov", type=float, default=DEFAULT_HFOV_DEG, help="Horizontal field of view (deg)")
    parser.add_argument("--width-m", type=float, default=REAL_DRONE_WIDTH_M, help="Real drone width in meters")
    parser.add_argument("--out", default="demo_output.jpg", help="Output path for annotated image or video")
    parser.add_argument("--show", action="store_true", help="Display interactive window")
    args = parser.parse_args()

    print("=" * 65)
    print(" VISION-BASED FORMATION POSE ESTIMATION: LIVE DEMONSTRATION")
    print("=" * 65)
    print(f"Model: {args.weights}")
    print(f"Source: {args.source}")
    print(f"Assumed Parameters: Drone Width = {args.width_m} m, Camera HFOV = {args.hfov}°\n")

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Trained model not found at: {args.weights}")

    model = YOLO(args.weights)
    source_str = args.source

    # Check if webcam or video
    is_cam = source_str.isdigit()
    is_video = source_str.endswith((".mp4", ".avi", ".mov", ".mkv")) or is_cam

    if is_video:
        cap = cv2.VideoCapture(int(source_str) if is_cam else source_str)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video stream: {source_str}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        focal_px = compute_focal_length(w, args.hfov)

        out_video_path = args.out if args.out.endswith(".mp4") else "demo_output.mp4"
        writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_count = 0
        print(f"Processing video stream ({w}x{h} @ {fps:.1f} FPS)...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ann, _, _ = process_frame(frame, model, focal_px, args.width_m)
            writer.write(ann)
            frame_count += 1
            if args.show:
                cv2.imshow("Formation Pose Estimation Live Demo", ann)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()
        print(f"Completed! Processed {frame_count} frames -> {out_video_path}")

    else:
        img_path = Path(source_str)
        if not img_path.exists():
            raise FileNotFoundError(f"Source image not found: {source_str}")

        frame = cv2.imread(str(img_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {source_str}")

        h, w = frame.shape[:2]
        focal_px = compute_focal_length(w, args.hfov)
        ann, detections, status = process_frame(frame, model, focal_px, args.width_m)

        cv2.imwrite(args.out, ann)
        print(f"Detections: {len(detections)}")
        for d in detections:
            print(f"  Drone #{d['id']}: 2D Box=({d['x1']:.1f}, {d['y1']:.1f}, {d['x2']:.1f}, {d['y2']:.1f}) | "
                  f"Estimated 3D Pose: (X={d['X_m']:+.2f}m, Y={d['Y_m']:+.2f}m, Z={d['Z_m']:.2f}m) | Conf={d['conf']:.2f}")
        print(f"Formation Status: {status}")
        print(f"\nSaved annotated demonstration to: {args.out}")

        if args.show:
            cv2.imshow("Formation Pose Estimation Live Demo", ann)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
