# Vision-Based Precision Pose Estimation for Autonomous Formation Flying
**Course:** UE24CS352A - Machine Learning Mini-Project  
**Team Members:** Manish Bala & Manasvi D R  
**Institution:** Department of Computer Science & Engineering, PES University  
**Repository:** [github.com/manishribil/Vision-Based-Pose-Estimation](https://github.com/manishribil/Vision-Based-Pose-Estimation)

---

## 1. Problem Statement & Motivation
Autonomous multi-Unmanned Aerial Vehicle (UAV) formation flight is critical for cooperative reconnaissance, disaster relief, payload transport, and perimeter surveillance. Traditional swarm positioning relies heavily on Global Navigation Satellite Systems (GNSS) and active RF inter-drone communication. However, in GNSS-denied environments (e.g., indoor flight, urban canyons, electronic jamming, or spoofing), these systems become entirely unreliable. 

This project implements an onboard, vision-based relative pose estimation and formation health monitoring pipeline. Using purely passive monocular visual data captured from a chase UAV, the system detects neighbouring drones, measures inter-vehicle relative spatial geometry in real time, computes distance and depth displacement, and classifies the swarm formation state without requiring active inter-drone radio communication.

---

## 2. Dataset Specifications
The pipeline is trained and benchmarked on the Roboflow **Multiple Drones** aerial formation dataset:
- **Target Platform:** Synchronized multi-drone flight of 3DR Solo quadcopters.
- **Physical Specifications:** Known motor-to-motor diagonal width $W = 0.46\text{ m}$.
- **Resolution & Modality:** Native $640 \times 640$ RGB imagery captured from chase-drone perspectives.
- **Test Benchmark Split:** 96 continuous trajectory test frames with 277 total annotated drone bounding boxes.
- **Environmental Dynamics:** High visual variance including cloud cover, variable lighting, sky-background contrast, scale variations (bounding box widths ranging from $8.4\text{ px}$ to $154.6\text{ px}$), and multi-drone occlusion.

---

## 3. Methodology & System Architecture

Our end-to-end framework bridges deep neural detection with geometric computer vision across 10 sequential phases:

```
[ Input Video Frame (640x640) ]
              │
              ▼
    [ Phase 1 & 2: YOLO11n Drone Detector ]
              │  ├── 2D Bounding Boxes: [x1, y1, x2, y2]
              │  └── Centroid Computation: cx = (x1+x2)/2, cy = (y1+y2)/2
              ▼
    [ Phase 3: Spatial Arrangement Engine ]
              │  └── Bearing Relationships (Left/Right, Above/Below)
              ▼
    [ Phase 5 & 6: Coordinate Normalization ]
              │  └── Resolution-Invariant Coordinates: x_norm = x / W_img, y_norm = y / H_img
              ▼
    [ Phase 7 & 8: Pairwise Euclidean Distances & Swarm Statistics ]
              │  └── All (N*(N-1))/2 inter-drone distances: dist = sqrt(dx^2 + dy^2)
              ▼
    [ Phase 10: Pinhole 3D Pose & Formation Health Tracking ]
              ├── 3D Distance Estimation: Z = (f * W) / w
              ├── Lateral & Elevation Offsets: X = (cx - c0)*Z / f,  Y = (cy - c0)*Z / f
              ├── Dimensionless Spacing: s = d_px / w_bar (drone-widths)
              ├── Formation State: Maintained (Green) / Too Close (Red) / Too Far (Orange)
              └── HUD & Bird's-Eye Top-Down Radar View Inset
```

### Key Mathematical Formulations:
1. **Centroid Extraction:**
   $$\begin{aligned} c_x = \frac{x_1 + x_2}{2}, \quad c_y = \frac{y_1 + y_2}{2} \end{aligned}$$
2. **Normalized Coordinates:**
   $$\begin{aligned} x_{\text{norm}} = \frac{c_x}{W_{\text{img}}}, \quad y_{\text{norm}} = \frac{c_y}{H_{\text{img}}} \end{aligned}$$
3. **Pinhole Monocular Distance Estimation:**
   $$\begin{aligned} Z = \frac{f \cdot W}{w} \end{aligned}$$
   Where $W = 0.46\text{ m}$ (physical drone width), $w = x_2 - x_1$ (detected box width in pixels), and $f$ is camera focal length in pixels:
   $$\begin{aligned} f = \frac{W_{\text{img}}}{2 \cdot \tan(\text{HFOV} / 2)} = \frac{640}{2 \cdot \tan(30^\circ)} \approx 554.26\text{ px} \end{aligned}$$
4. **Dimensionless Formation Spacing:**
   $$\begin{aligned} s = \frac{\sqrt{(c_{x,B} - c_{x,A})^2 + (c_{y,B} - c_{y,A})^2}}{\frac{w_A + w_B}{2}} \quad \text{(measured in drone-widths)} \end{aligned}$$
   - **Maintained:** $3.0 \le s \le 10.0$ drone-widths
   - **Too Close (Collision Hazard):** $s < 3.0$ drone-widths
   - **Too Far (Formation Breakdown):** $s > 10.0$ drone-widths

---

## 4. Quantitative Experimental Results

### Object Detection Performance (Test Split):
| Metric | Score | Percentage | Analysis |
|---|---:|---:|---|
| **Precision (P)** | 0.9963 | **99.63%** | Minimal false alarms in operating domain |
| **Recall (R)** | 0.9951 | **99.51%** | 99.5% of drones detected across all frames |
| **mAP@50** | 0.9936 | **99.36%** | Excellent overlap at standard IoU threshold |
| **mAP@50-95** | 0.8512 | **85.12%** | High strictness across multi-scale drones |

### Formation State Distribution across 96 Evaluation Frames:
- **Maintained (Nominal):** 44 frames (45.8%)
- **Outside Threshold (Too Far):** 34 frames (35.4%)
- **Single Drone Detected:** 12 frames (12.5%)
- **Too Close (Collision Hazard):** 6 frames (6.3%)
- **Nearest-Neighbour Spacing:** Median $5.9$ drone-widths (IQR: $4.1 - 9.4$)
- **Physical Estimated Range:** Median $8.5\text{ m}$ (Min: $1.6\text{ m}$, Max: $30.2\text{ m}$)

---

## 5. Depth Error & Sensitivity Analysis

### Comparison Against Ground-Truth Bounding Boxes (272 Matched Pairs, Mean IoU = 0.915):
- **Mean Absolute Error (MAE):** $0.252\text{ m}$
- **Root Mean Squared Error (RMSE):** $0.360\text{ m}$
- **Mean Absolute Percentage Error (MAPE):** $2.5\%$
- **90th Percentile Error:** $5.2\%$

### Mathematical Sensitivity Derivation:
Differentiating the pinhole equation $Z = \frac{f \cdot W}{w}$ with respect to pixel box width $w$:
$$\frac{dZ}{dw} = -\frac{f \cdot W}{w^2} = -\frac{Z}{w}$$
Taking the relative fractional error:
$$\left|\frac{\delta Z}{Z}\right| = \left|\frac{\delta w}{w}\right|$$

**Key Physical Finding:**  
The percentage distance error is strictly equal to the percentage box-width error. For a close drone ($w = 120\text{ px}$), a $1\text{ px}$ detector error yields only $0.83\%$ distance error ($\approx 0.01\text{ m}$). However, for a distant drone ($w = 10\text{ px}$), the same $1\text{ px}$ error generates a $10.0\%$ distance error ($\approx 2.5\text{ m}$). Bounding box regression quality is therefore the primary sensitivity bottleneck for monocular range estimation.

---

## 6. Edge Case Stress Testing & Failure Modes
Using `tools/edge_case_tests.py`, we subjected the detector to systematic synthetic and out-of-distribution stress tests:
1. **Motion Blur:** Minor flight vibration ($K=15$ blur kernel) reduces confidence to $0.11$. Severe blur ($K=35$) causes total detection dropout ($0$ boxes).
2. **Illumination Scaling:** At $50\%$ and $20\%$ brightness, the detector retains strong recall ($\text{conf} \approx 0.35$) thanks to silhouette contrast against the open sky.
3. **Boundary Truncation:** Drones partially exiting frame boundaries ($35\%$ cropping) truncate the bounding box width $w$, which artificially inflates estimated distance $Z$.
4. **Out-of-Distribution Urban Scene (`bus.jpg`):** When tested on an urban street scene without drones, the model triggered a false positive ($\text{conf} = 0.47$) on vehicle window textures.
5. **Airframe Morphology Bias:** A close-up white commercial drone (`drone.jpg`) triggered split/partial bounding boxes because the model was trained exclusively on dark 3DR Solo airframes against sky.

---

## 7. Conclusions & Future Work
- **Conclusions:** We successfully developed, implemented, and verified an end-to-end 10-phase vision-based pose estimation pipeline. Dimensionless drone-width spacing provides an immediate, calibration-free metric for collision avoidance and formation monitoring. With calibrated camera optics, monocular distance estimation achieves sub-meter accuracy ($\text{MAE} = 0.252\text{ m}$).
- **Future Work:**
  1. Integrate Extended Kalman Filtering (EKF) to fuse optical flow and smooth depth jitter.
  2. Implement stereo vision or visual odometry (VO) to remove single-camera physical size dependency.
  3. Expand dataset diversity with hard-negative background mining to eliminate urban false positives.

---

## 8. Individual Contributions
- **Manish Bala:**
  - Selected, configured, and trained the YOLO11n object detector (Phase 1, Phase 4).
  - Designed the 2D bounding box and centroid extraction pipeline (Phase 2).
  - Developed relative bearing and horizontal/vertical spatial arrangement logic (Phase 3).
  - Implemented image coordinate normalization (Phase 6) and initial pairwise distance calculation (Phase 7).
  - Managed primary git repository setup and baseline documentation.
- **Manasvi D R:**
  - Formulated pinhole monocular 3D distance and lateral displacement algorithms (Phase 10).
  - Developed the dimensionless nearest-neighbor drone-width spacing metric and dynamic formation health classifier (Maintained / Too Close / Outside Threshold).
  - Built the live telemetry HUD overlay and bird's-eye top-down radar projection (Phase 10 & `demo.py`).
  - Derived mathematical error sensitivity $\left|\frac{\delta Z}{Z}\right| = \left|\frac{\delta w}{w}\right|$ and evaluated ground-truth error metrics (MAE, RMSE, MAPE).
  - Developed the automated edge-case robustness test suite (`tools/edge_case_tests.py`).
  - Created the physical camera calibration and error validation tool (`tools/manasvi_distance_viz.py`).
  - Implemented the master pipeline verification harness (`verify_pipeline.py`) and generated project PDF deliverables.
