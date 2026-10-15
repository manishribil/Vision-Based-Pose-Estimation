"""
Manasvi's part: approximate distance + relative position + overlay + error analysis.

Usage examples
--------------
# 1) Calibrate focal length (photo of the drone taken at a KNOWN distance)
python manasvi_distance_viz.py calibrate --image near_2m.jpg --distance 2.0

# 2) Run on an image or video with the calibrated focal length
python manasvi_distance_viz.py run --source drone_jpg.jpeg --focal 1450

# 3) Error analysis from a CSV of (true_m, estimated_m) pairs
python manasvi_distance_viz.py errors --csv measurements.csv
"""
import argparse
import csv
import os

import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS = "runs/detect/runs/drone/formation_v1/weights/best.pt"
# ASSUMPTION TO VALIDATE: real width of the drone as seen by the box.
# 3DR Solo is 0.46 m motor-to-motor diagonal; the box also includes
# propellers, so the true visible width is a bit larger. Measure YOUR drone.
REAL_WIDTH_M = 0.46
TARGET_DISTANCE_M = 5.0     # desired formation distance
TOLERANCE_M = 1.0           # +/- allowed
CONF = 0.25


# ---------- core maths ----------
def estimate_distance(focal_px, real_width_m, width_px):
    """Pinhole approximation: Z = f * W / w"""
    return (focal_px * real_width_m) / width_px


def calibrate_focal(width_px, known_distance_m, real_width_m):
    """Invert the formula: f = w * Z / W"""
    return (width_px * known_distance_m) / real_width_m


def relative_position(cx, img_w):
    """Left / center / right using thirds of the image width."""
    if cx < img_w / 3:
        return "left"
    if cx > 2 * img_w / 3:
        return "right"
    return "center"


def formation_status(dist_m):
    ok = abs(dist_m - TARGET_DISTANCE_M) <= TOLERANCE_M
    return "maintained" if ok else "outside threshold"


# ---------- detection ----------
def detect_best(model, frame):
    """Return highest-confidence box (x1,y1,x2,y2,conf) or None."""
    res = model.predict(frame, conf=CONF, verbose=False)[0]
    if len(res.boxes) == 0:
        return None
    i = int(res.boxes.conf.argmax())
    x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()
    return x1, y1, x2, y2, float(res.boxes.conf[i])


# ---------- visualization ----------
def draw_overlay(frame, det, focal_px):
    h, w = frame.shape[:2]
    if det is None:
        cv2.putText(frame, "NO DRONE DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame, None

    x1, y1, x2, y2, conf = det
    bw = x2 - x1
    dist = estimate_distance(focal_px, REAL_WIDTH_M, bw)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    pos = relative_position(cx, w)
    status = formation_status(dist)
    color = (0, 200, 0) if status == "maintained" else (0, 0, 255)

    # box + centre point
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.circle(frame, (int(cx), int(cy)), 5, color, -1)
    # image centre crosshair + offset line
    cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255),
                   cv2.MARKER_CROSS, 30, 2)
    cv2.line(frame, (w // 2, h // 2), (int(cx), int(cy)), (255, 255, 0), 1)
    # left/center/right guide lines
    for xv in (w // 3, 2 * w // 3):
        cv2.line(frame, (xv, 0), (xv, h), (120, 120, 120), 1)

    lines = [
        "DRONE DETECTED",
        f"Confidence: {conf:.2f}",
        f"Box: ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})  w={bw:.0f}px",
        f"Estimated distance: {dist:.2f} m",
        f"Relative position: {pos}",
        f"Formation: {status}",
    ]
    # background panel for readability
    cv2.rectangle(frame, (10, 10), (520, 20 + 28 * len(lines)), (0, 0, 0), -1)
    for k, t in enumerate(lines):
        cv2.putText(frame, t, (20, 36 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color if k else (255, 255, 255), 2)
    return frame, dict(conf=conf, width_px=bw, dist=dist, pos=pos, status=status)


# ---------- commands ----------
def cmd_calibrate(a):
    model = YOLO(WEIGHTS)
    frame = cv2.imread(a.image)
    det = detect_best(model, frame)
    if det is None:
        raise SystemExit("No drone detected in calibration image.")
    bw = det[2] - det[0]
    f = calibrate_focal(bw, a.distance, REAL_WIDTH_M)
    print(f"Box width = {bw:.1f}px at {a.distance} m -> focal length = {f:.1f} px")
    print(f"Use:  --focal {f:.0f}")


def cmd_run(a):
    model = YOLO(WEIGHTS)
    os.makedirs("outputs", exist_ok=True)
    ext = os.path.splitext(a.source)[1].lower()
    is_video = ext in (".mp4", ".avi", ".mov", ".mkv")

    if not is_video:
        frame = cv2.imread(a.source)
        out, info = draw_overlay(frame, detect_best(model, frame), a.focal)
        path = os.path.join("outputs", "result_" + os.path.basename(a.source))
        cv2.imwrite(path, out)
        print("Saved", path, info)
        return

    cap = cv2.VideoCapture(a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    size = (int(cap.get(3)), int(cap.get(4)))
    writer = cv2.VideoWriter("outputs/result_video.mp4",
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    log = open("outputs/log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["frame", "conf", "width_px", "dist_m", "position", "status"])
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out, info = draw_overlay(frame, detect_best(model, frame), a.focal)
        if info:
            wr.writerow([n, f"{info['conf']:.2f}", f"{info['width_px']:.0f}",
                         f"{info['dist']:.2f}", info["pos"], info["status"]])
        writer.write(out)
        n += 1
    cap.release(); writer.release(); log.close()
    print(f"Processed {n} frames -> outputs/result_video.mp4, outputs/log.csv")


def cmd_errors(a):
    import matplotlib.pyplot as plt
    true, est = [], []
    with open(a.csv) as f:
        for row in csv.DictReader(f):
            true.append(float(row["true_m"])); est.append(float(row["estimated_m"]))
    true, est = np.array(true), np.array(est)
    err = est - true
    pct = 100 * np.abs(err) / true
    print(f"MAE  = {np.mean(np.abs(err)):.2f} m")
    print(f"RMSE = {np.sqrt(np.mean(err**2)):.2f} m")
    print(f"Mean % error = {pct.mean():.1f}%")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(true, true, "k--", label="ideal")
    ax[0].scatter(true, est, c="tab:blue", label="estimated")
    ax[0].set_xlabel("True distance (m)"); ax[0].set_ylabel("Estimated (m)")
    ax[0].legend(); ax[0].set_title("Estimated vs true")
    ax[1].bar(range(len(pct)), pct)
    ax[1].set_xticks(range(len(pct))); ax[1].set_xticklabels(true)
    ax[1].set_xlabel("True distance (m)"); ax[1].set_ylabel("% error")
    ax[1].set_title("Percentage error")
    plt.tight_layout(); plt.savefig("outputs_error_analysis.png", dpi=150)
    print("Saved outputs_error_analysis.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--image", required=True)
    c.add_argument("--distance", type=float, required=True)
    r = sub.add_parser("run")
    r.add_argument("--source", required=True)
    r.add_argument("--focal", type=float, required=True)
    e = sub.add_parser("errors")
    e.add_argument("--csv", required=True)
    a = p.parse_args()
    {"calibrate": cmd_calibrate, "run": cmd_run, "errors": cmd_errors}[a.cmd](a)
