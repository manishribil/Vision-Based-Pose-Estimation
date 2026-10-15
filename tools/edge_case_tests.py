"""
Manasvi - Step 2: detection quality and edge cases.

Put this next to best.pt and your test images, then:
    python edge_case_tests.py drone_jpg.jpeg [more_images ...]

For each image it runs best.pt on the original plus synthetic edge cases:
  far_X     drone shrunk to X% of the frame (simulates larger distance)
  blur_K    motion blur of kernel size K
  dark_X    brightness scaled to X
  cut_*     drone partly outside the frame (crop)
Saves annotated images in edge_cases/ and a summary in edge_cases/summary.csv.
Look in summary.csv for rows with detected=0 or low conf = your failure cases.
"""
import csv
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("runs/detect/runs/drone/formation_v1/weights/best.pt")
os.makedirs("edge_cases", exist_ok=True)


def far(img, pct):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * pct), int(h * pct)))
    canvas = np.full_like(img, 128)
    y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return canvas


def blur(img, k):
    kern = np.zeros((k, k)); kern[k // 2, :] = 1 / k
    return cv2.filter2D(img, -1, kern)


def dark(img, f):
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)


def cut(img, side):
    h, w = img.shape[:2]
    if side == "left":
        return img[:, int(w * 0.35):]
    if side == "right":
        return img[:, :int(w * 0.65)]
    return img[int(h * 0.35):, :]  # bottom


def variants(img):
    yield "original", img
    for p in (0.5, 0.25, 0.12, 0.06):
        yield f"far_{int(p * 100)}", far(img, p)
    for k in (15, 35):
        yield f"blur_{k}", blur(img, k)
    for f in (0.5, 0.2):
        yield f"dark_{int(f * 100)}", dark(img, f)
    for s in ("left", "right", "top"):
        yield f"cut_{s}", cut(img, s)


rows = []
for path in sys.argv[1:]:
    base = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path)
    if img is None:
        print("cannot read", path); continue
    for name, v in variants(img):
        res = model.predict(v, conf=0.10, verbose=False)[0]  # low conf to see weak hits
        n = len(res.boxes)
        best = float(res.boxes.conf.max()) if n else 0.0
        wpx = 0.0
        if n:
            x1, _, x2, _ = res.boxes.xyxy[int(res.boxes.conf.argmax())].tolist()
            wpx = x2 - x1
        cv2.imwrite(f"edge_cases/{base}_{name}.jpg", res.plot())
        rows.append([base, name, int(n > 0), n, f"{best:.2f}", f"{wpx:.0f}"])
        print(f"{base:20s} {name:10s} detected={int(n>0)} conf={best:.2f} w={wpx:.0f}px")

with open("edge_cases/summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "case", "detected", "num_boxes", "best_conf", "box_width_px"])
    w.writerows(rows)
print("\nSaved edge_cases/summary.csv and annotated images.")
