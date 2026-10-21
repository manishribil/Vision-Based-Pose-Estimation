# Presentation Slide Deck & Viva Q&A Guide
## Vision-Based Precision Pose Estimation for Autonomous Formation Flying
**Course:** UE24CS352A - Machine Learning Mini-Project  
**Team Members:** Manish Bala & Manasvi D R  
**Faculty Evaluation Date:** October 2026  

---

### Slide 1: Title & Overview
- **Title:** Vision-Based Precision Pose Estimation for Autonomous Formation Flying
- **Context:** UE24CS352A Machine Learning Mini-Project Review
- **Presenters:** Manish Bala & Manasvi D R
- **Key Objective:** Real-time, passive 2D/3D relative spatial tracking and formation health monitoring for GNSS-denied multi-drone swarms using lightweight edge computer vision.

---

### Slide 2: Problem Statement & Motivation
- **The Challenge:** Multi-UAV swarms rely heavily on GPS and inter-drone RF telemetry.
  - GPS is prone to jamming, spoofing, and multipath denial indoors or in urban canyons.
  - Radio communication introduces RF emissions, packet latency, and bandwidth bottlenecks.
- **The Solution:** Monocular computer vision on an onboard chase camera:
  - 100% passive sensing (zero RF emission).
  - High-rate visual telemetry (bearing, distance, lateral offset).
  - Dimensionless formation health tracking requiring zero initial camera calibration.

---

### Slide 3: System Architecture (10-Phase Pipeline)
- **Phase 1 & 2:** YOLO11n object detector & centroid extraction ($c_x, c_y$).
- **Phase 3:** Relative spatial bearing engine ($\Delta x, \Delta y$, Left/Right, Above/Below).
- **Phase 4:** Formal model evaluation on test split (Precision, Recall, mAP).
- **Phase 5 & 6:** Annotated visual overlays & coordinate normalization ($x/W, y/H$).
- **Phase 7 & 8:** Pairwise Euclidean separation & image-level swarm statistics.
- **Phase 9:** Swarm density distribution visualization.
- **Phase 10:** Pinhole monocular 3D depth ($Z = f \cdot W / w$), nearest-neighbor spacing in drone-widths, HUD overlay, and top-down radar view.

---

### Slide 4: Mathematical Formulations
1. **Pinhole Camera Distance Estimation:**
   $$Z = \frac{f \cdot W}{w}$$
   Where $W = 0.46\text{ m}$ (physical drone width), $w = x_2 - x_1$ (pixel box width), $f = 554\text{ px}$.
2. **Dimensionless Nearest-Neighbor Spacing:**
   $$s = \frac{d_{\text{px}}}{\bar{w}_{\text{px}}} = \frac{\sqrt{\Delta x^2 + \Delta y^2}}{(w_A + w_B) / 2} \quad \text{(drone-widths)}$$
   - **Maintained:** $3.0 \le s \le 10.0$ drone-widths
   - **Too Close (Collision Hazard):** $s < 3.0$ drone-widths
   - **Too Far (Formation Breakdown):** $s > 10.0$ drone-widths

---

### Slide 5: Detection Performance & Benchmark Results
| Metric | Score | Percentage | Significance |
|---|---:|---:|---|
| **Precision** | 0.9963 | **99.63%** | Minimal false alarms |
| **Recall** | 0.9951 | **99.51%** | 99.5% of drones detected across all frames |
| **mAP@50** | 0.9936 | **99.36%** | Outstanding localization at standard IoU |
| **mAP@50-95** | 0.8512 | **85.12%** | High strictness across small and large scales |

---

### Slide 6: Depth Error & Sensitivity Analysis
- **Ground-Truth Comparison (272 matched detections, mean IoU = 0.915):**
  - **MAE:** $0.252\text{ m}$ | **RMSE:** $0.360\text{ m}$
  - **Mean Percentage Error:** $2.5\%$ | **90th Percentile Error:** $5.2\%$
- **Mathematical Sensitivity:**
  $$\frac{dZ}{dw} = -\frac{Z}{w} \implies \left|\frac{\delta Z}{Z}\right| = \left|\frac{\delta w}{w}\right|$$
  - Near drone ($w = 120\text{ px}$): $1\text{ px}$ jitter $\to 0.83\%$ error.
  - Far drone ($w = 10\text{ px}$): $1\text{ px}$ jitter $\to 10.0\%$ error.

---

### Slide 7: Stress Testing & Failure Modes
Empirical testing using `tools/edge_case_tests.py`:
1. **Motion Blur:** $K=15$ drops confidence to $0.11$; $K=35$ leads to total detection loss.
2. **Illumination:** Retains robust detection ($\text{conf} \approx 0.35$) even under $20\%$ dark scaling.
3. **Boundary Truncation:** $35\%$ cropping truncates box width, artificially inflating estimated distance.
4. **Out-of-Distribution Urban Clutter (`bus.jpg`):** Detector produced a false positive ($\text{conf} = 0.47$) on vehicle window contours.
5. **Morphological Bias (`drone.jpg`):** White close-up drone caused split boxes due to dark 3DR Solo training bias.

---

### Slide 8: Live Demonstration & End-to-End Verification
- **Automated Verification:** `python verify_pipeline.py` passes all 10 phases in 1.38s.
- **Interactive Live Demo:** `python demo.py --source test_images/drone.jpg`
  - Real-time HUD telemetry.
  - Multi-drone distance linking.
  - Bird's-eye top-down radar view.

---

### Slide 9: Individual Contributions
- **Manish Bala:**
  - YOLO11n model selection and fine-tuning.
  - 2D centroid extraction and bearing analysis (Phases 1-3).
  - Coordinate normalization & pairwise distances (Phases 6-7).
- **Manasvi D R:**
  - Pinhole 3D distance formulation & lateral offsets (Phase 10).
  - Dimensionless drone-width spacing & formation health state classifier.
  - Telemetry HUD & top-down radar map projection (`demo.py`).
  - Mathematical sensitivity analysis & ground-truth evaluation.
  - Automated edge-case testing suite & camera calibration tools.
  - Master verification harness & 2-page project report PDF.

---

## Viva & Demonstration Q&A Preparation

### Q1: Why use YOLO11n rather than a larger model like YOLO11x or Faster R-CNN?
**Answer:** Autonomous formation flying requires onboard inference on resource-constrained embedded edge computers (such as NVIDIA Jetson Orin Nano or Raspberry Pi 5). YOLO11n has only ~2.6M parameters and ~6.5 GFLOPs, achieving >60 FPS inference while still delivering 99.36% mAP@50 on our test split. Larger models would add latency that could cause collision avoidance delays.

### Q2: Why measure formation spacing in "drone-widths" rather than meters in Phase 10?
**Answer:** Estimating physical meters requires knowing both the exact physical target dimension ($W$) and the camera's optical focal length ($f$). If the drone operates with an uncalibrated camera or a zoom lens, meter values drift. However, the ratio $s = d_{\text{px}} / \bar{w}_{\text{px}}$ is dimensionless; since both pixel separation and box width scale proportionally with distance, $s$ is scale-invariant and calibration-free, providing an immediate collision warning even without camera calibration.

### Q3: How do you handle detection jitter when a drone is far away?
**Answer:** As shown in our sensitivity derivation $\left|\frac{\delta Z}{Z}\right| = \left|\frac{\delta w}{w}\right|$, when a drone is far away ($w \approx 10\text{ px}$), a $1\text{ px}$ bounding box jitter yields a $10\%$ depth jump. In future work, this will be filtered using an Extended Kalman Filter (EKF) with a constant-velocity kinematic motion model to smooth temporal jitter.

### Q4: What happened when the detector was tested on `bus.jpg`?
**Answer:** The detector exhibited an out-of-distribution false positive on the bus window contours with a confidence of 0.47. This occurs because the training dataset consisted entirely of drones against the sky, with no hard-negative non-drone urban backgrounds. In production, fine-tuning with hard-negative background images is required to suppress spurious background activations.
