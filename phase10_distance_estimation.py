"""
Manasvi - distance estimation, relative-position visualization and error analysis
built on top of Manish's detections (drone_positions.csv).

Run from the repo root:
    python3 phase10_distance_estimation.py \
        --positions runs/phase2_position/drone_positions.csv \
        --images "data/Multiple Drones.v1i.yolov11/test/images" \
        --labels "data/Multiple Drones.v1i.yolov11/test/labels" \
        --out results/distance

Method
------
Z = f * W / w          (pinhole approximation; w = box width in pixels)
X = (cx - c0) * Z / f  (lateral offset),  Y = (cy - c0) * Z / f
Formation status uses *separation in drone-widths* (pixel separation / box width),
which needs no camera calibration. Metres need W and f, which are ASSUMPTIONS here.
"""
import argparse
import math
import os
import re

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMG = 640  # dataset images are 640 x 640


def focal_px(hfov_deg):
    return (IMG / 2) / math.tan(math.radians(hfov_deg) / 2)


def frame_no(name):
    m = re.search(r"mp4-(\d+)_", name)
    return int(m.group(1)) if m else 0


# ------------------------------------------------------------------ metrics
def enrich(df, f, W):
    df = df.copy()
    df["w_px"] = df.x2 - df.x1
    df["Z_m"] = f * W / df.w_px
    df["X_m"] = (df.center_x - IMG / 2) * df.Z_m / f
    df["Y_m"] = (df.center_y - IMG / 2) * df.Z_m / f
    nn_w, nn_id, nn_m = [], [], []
    for _, g in df.groupby("image", sort=False):
        c = g[["center_x", "center_y"]].values
        w = g.w_px.values
        P = g[["X_m", "Y_m", "Z_m"]].values
        for i in range(len(g)):
            if len(g) < 2:
                nn_w.append(np.nan); nn_id.append(-1); nn_m.append(np.nan)
                continue
            d_px = np.hypot(*(c - c[i]).T)
            sep_w = d_px / ((w + w[i]) / 2)
            sep_w[i] = np.inf
            j = int(np.argmin(sep_w))
            nn_w.append(sep_w[j]); nn_id.append(int(g.drone_id.values[j]))
            nn_m.append(float(np.linalg.norm(P[i] - P[j])))
    df["nn_id"] = nn_id
    df["nn_sep_widths"] = nn_w
    df["nn_sep_m"] = nn_m
    return df


def frame_status(g, lo, hi):
    if len(g) < 2:
        return "single drone", (200, 200, 200)
    s = g.nn_sep_widths.values
    if (s < lo).any():
        return "too close", (0, 0, 255)
    if (s > hi).any():
        return "outside threshold (too far)", (0, 140, 255)
    return "maintained", (0, 200, 0)


# ------------------------------------------------------------ visualization
def topdown_map(g, size=190):
    """Small bird's-eye inset: x = lateral offset, y = depth (m). Camera at bottom centre."""
    pad = 22
    m = np.full((size, size, 3), 30, np.uint8)
    zmax = max(g.Z_m.max() * 1.15, 1.0)
    xmax = max(np.abs(g.X_m).max() * 1.3, 1.0)
    cv2.rectangle(m, (0, 0), (size - 1, size - 1), (120, 120, 120), 1)
    cam = (size // 2, size - 8)
    cv2.drawMarker(m, cam, (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 10, 1)
    for _, r in g.iterrows():
        px = int(size / 2 + r.X_m / xmax * (size / 2 - pad))
        py = int(size - 14 - r.Z_m / zmax * (size - 14 - pad))
        cv2.circle(m, (px, py), 5, (0, 220, 255), -1)
        cv2.putText(m, str(int(r.drone_id)), (px + 6, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(m, "top view", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(m, f"depth to {zmax:.0f} m", (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    return m


def draw_frame(img, g, lo, hi, scale=1.5):
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    s = scale
    status, col = frame_status(g, lo, hi)
    by_id = {int(r.drone_id): r for _, r in g.iterrows()}
    for _, r in g.iterrows():
        if r.nn_id in by_id:
            o = by_id[r.nn_id]
            cv2.line(img, (int(r.center_x * s), int(r.center_y * s)),
                     (int(o.center_x * s), int(o.center_y * s)), (255, 255, 0), 1)
    for _, r in g.iterrows():
        x1, y1, x2, y2 = [int(v * s) for v in (r.x1, r.y1, r.x2, r.y2)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"#{int(r.drone_id)} {r.Z_m:.1f}m"
        cv2.putText(img, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    lines = [f"DRONES DETECTED: {len(g)}", f"Formation: {status}"]
    if len(g) > 1:
        lines.append(f"Neighbour spacing: {g.nn_sep_widths.min():.1f} - {g.nn_sep_widths.max():.1f} drone-widths")
        lines.append(f"Depth range: {g.Z_m.min():.1f} - {g.Z_m.max():.1f} m (assumed camera)")
    h = 12 + 20 * len(lines)
    cv2.rectangle(img, (0, 0), (430, h), (0, 0, 0), -1)
    for k, t in enumerate(lines):
        cv2.putText(img, t, (8, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col if k == 1 else (255, 255, 255), 1)
    mp = topdown_map(g)
    H, Wd = img.shape[:2]
    img[H - mp.shape[0] - 6:H - 6, Wd - mp.shape[1] - 6:Wd - 6] = mp
    cv2.putText(img, os.path.basename(g.image.iloc[0]).split("_jpg")[0][-18:], (8, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return img


# ------------------------------------------------------------ error analysis
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def load_gt(label_dir, image):
    path = os.path.join(label_dir, os.path.splitext(image)[0] + ".txt")
    G = []
    if os.path.exists(path):
        for l in open(path).read().splitlines():
            if l.strip():
                _, x, y, w, h = map(float, l.split()[:5])
                G.append(((x - w / 2) * IMG, (y - h / 2) * IMG, (x + w / 2) * IMG, (y + h / 2) * IMG))
    return G


def error_analysis(df, label_dir, f, W, out):
    rows = []
    for img, g in df.groupby("image"):
        G = load_gt(label_dir, img)
        used = set()
        for _, r in g.sort_values("confidence", ascending=False).iterrows():
            b = (r.x1, r.y1, r.x2, r.y2)
            best, bi = -1, 0
            for j, gt in enumerate(G):
                if j in used:
                    continue
                v = iou(b, gt)
                if v > bi:
                    bi, best = v, j
            if bi >= 0.5:
                used.add(best)
                wg = G[best][2] - G[best][0]
                rows.append(dict(image=img, w_det=r.w_px, w_gt=wg, iou=bi))
    e = pd.DataFrame(rows)
    e["Z_det"] = f * W / e.w_det
    e["Z_gt"] = f * W / e.w_gt
    e["err_m"] = e.Z_det - e.Z_gt
    e["err_pct"] = 100 * e.err_m / e.Z_gt
    mae, rmse = e.err_m.abs().mean(), math.sqrt((e.err_m ** 2).mean())
    mape = e.err_pct.abs().mean()
    p90 = e.err_pct.abs().quantile(0.9)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].hist(e.err_pct, bins=30, color="tab:blue", edgecolor="white")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_xlabel("distance error (%)  [detected box vs ground-truth box]")
    ax[0].set_ylabel("count"); ax[0].set_title(f"Distance error from box-width error (n={len(e)})")
    ax[1].scatter(e.w_gt, e.err_pct.abs(), s=14, alpha=0.6)
    ax[1].set_xlabel("ground-truth box width (px)"); ax[1].set_ylabel("|error| (%)")
    ax[1].set_title("Small boxes -> larger distance error")
    ww = np.linspace(8, 160, 200)
    ax[2].plot(ww, 100 / ww, label="+1 px width error")
    ax[2].plot(ww, 200 / ww, label="+2 px width error")
    ax[2].set_xscale("log"); ax[2].set_xlabel("box width (px)")
    ax[2].set_ylabel("distance error (%)"); ax[2].legend(); ax[2].grid(alpha=0.3)
    ax[2].set_title("Sensitivity: error ~ (pixel error) / width")
    plt.tight_layout(); plt.savefig(os.path.join(out, "error_analysis.png"), dpi=150); plt.close()
    e.to_csv(os.path.join(out, "error_details.csv"), index=False)
    return e, dict(n=len(e), mae=mae, rmse=rmse, mape=mape, p90=p90,
                   mean_iou=e.iou.mean(), median_w=e.w_gt.median())


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default="results/formation")
    ap.add_argument("--width-m", type=float, default=0.46, help="real drone width W (3DR Solo motor-to-motor = 0.46 m)")
    ap.add_argument("--hfov", type=float, default=60.0, help="ASSUMED horizontal field of view (deg)")
    ap.add_argument("--min-sep", type=float, default=3.0, help="too-close threshold (drone-widths)")
    ap.add_argument("--max-sep", type=float, default=10.0, help="too-far threshold (drone-widths)")
    ap.add_argument("--fps", type=float, default=3.0)
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "frames"), exist_ok=True)
    f = focal_px(a.hfov)
    df = enrich(pd.read_csv(a.positions), f, a.width_m)
    df["frame"] = df.image.map(frame_no)
    df = df.sort_values(["frame", "drone_id"]).reset_index(drop=True)

    statuses, written = [], []
    for img_name, g in df.groupby("image", sort=False):
        img = cv2.imread(os.path.join(a.images, img_name))
        if img is None:
            continue
        out = draw_frame(img, g, a.min_sep, a.max_sep)
        p = os.path.join(a.out, "frames", f"{frame_no(img_name):05d}.jpg")
        cv2.imwrite(p, out)
        written.append(p)
        statuses.append((frame_no(img_name), frame_status(g, a.min_sep, a.max_sep)[0], len(g)))
    if written:
        h, w = cv2.imread(written[0]).shape[:2]
        vw = cv2.VideoWriter(os.path.join(a.out, "formation_demo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
        for p in written:
            vw.write(cv2.imread(p))
        vw.release()

    df.drop(columns=["frame"]).to_csv(os.path.join(a.out, "metrics.csv"), index=False)

    st = pd.DataFrame(statuses, columns=["frame", "status", "n"])
    counts = st.status.value_counts()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    counts.plot.bar(ax=ax[0], color=["tab:green", "tab:orange", "tab:red", "tab:gray"][:len(counts)])
    ax[0].set_title("Formation status per frame"); ax[0].set_ylabel("frames"); ax[0].tick_params(axis="x", rotation=20)
    v = df.nn_sep_widths.dropna()
    ax[1].hist(v, bins=40, color="tab:blue", edgecolor="white")
    ax[1].axvline(a.min_sep, color="r", ls="--"); ax[1].axvline(a.max_sep, color="r", ls="--")
    ax[1].set_xlabel("nearest-neighbour spacing (drone-widths)"); ax[1].set_title("Spacing distribution (red = thresholds)")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "summary.png"), dpi=150); plt.close()

    lines = [f"Assumed: W = {a.width_m} m, HFOV = {a.hfov} deg -> f = {f:.0f} px (image {IMG}x{IMG})",
             f"Frames: {len(st)}   Detections: {len(df)}",
             f"Detected width px: median {df.w_px.median():.1f}, min {df.w_px.min():.1f}, max {df.w_px.max():.1f}",
             f"Estimated distance (m): median {df.Z_m.median():.1f}, min {df.Z_m.min():.1f}, max {df.Z_m.max():.1f}",
             f"Nearest-neighbour spacing (drone-widths): median {v.median():.1f}, IQR {v.quantile(.25):.1f}-{v.quantile(.75):.1f}",
             f"Nearest-neighbour spacing (m, 3D): median {df.nn_sep_m.median():.2f}",
             "Frame status counts: " + ", ".join(f"{k}={c}" for k, c in counts.items())]
    multi = df.groupby("image").filter(lambda g: len(g) > 1).groupby("image").w_px.agg(lambda s: s.max() / s.min())
    lines.append(f"Within-frame width ratio (max/min), multi-drone frames: median {multi.median():.2f}")

    if a.labels:
        e, s = error_analysis(df, a.labels, f, a.width_m, a.out)
        lines += ["", "Distance error caused by box-width error (detected box vs ground-truth box):",
                  f"  matched boxes n={s['n']}, mean IoU={s['mean_iou']:.3f}, median GT width={s['median_w']:.1f}px",
                  f"  MAE = {s['mae']:.3f} m, RMSE = {s['rmse']:.3f} m, mean |error| = {s['mape']:.1f}%, 90th pct = {s['p90']:.1f}%"]

    txt = "\n".join(lines)
    open(os.path.join(a.out, "report_numbers.txt"), "w").write(txt)
    print(txt)
    print("\nSaved to", a.out)


if __name__ == "__main__":
    main()
