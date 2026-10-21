# Vision-Based Pose Estimation

## Project Overview

This project uses computer vision to detect multiple drones in images, estimate their 2D image positions, and analyze their relative spatial arrangement.

The system uses a YOLO11n object detection model and Python-based image processing to extract drone positions, calculate pairwise distances, and visualize formation patterns.

## Objectives

- Detect drones in test images using a trained YOLO11n model.
- Extract drone bounding boxes and centre coordinates.
- Analyze relative horizontal and vertical positions.
- Calculate pairwise distances in image coordinates.
- Normalize image coordinates for formation mapping.
- Evaluate detection performance and visualize formation statistics.

## Methodology

1. Train the YOLO11n drone detector.
2. Extract 2D drone centre coordinates.
3. Calculate relative positions and pairwise pixel distances.
4. Evaluate the model on the test dataset.
5. Generate annotated formation visualizations.
6. Normalize image coordinates.
7. Calculate normalized pairwise distances.
8. Summarize formation statistics.
9. Visualize distance distributions.

## Test-Set Results

| Metric | Result |
|---|---:|
| Precision | 99.63% |
| Recall | 99.51% |
| mAP@50 | 99.36% |
| mAP@50–95 | 85.12% |

These results apply to the project's current test split and should not be interpreted as a guarantee of performance in all real-world environments.

## Technology Stack

- Python
- YOLO11n and Ultralytics
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Installation and Execution

### Prerequisites

- Python 3.10+ (PyTorch, Ultralytics, OpenCV, NumPy, Pandas, Matplotlib)
- Git

### Fast One-Command Verification

To run and verify the entire project pipeline from Phase 1 to Phase 10 with a detailed report:

```bash
python verify_pipeline.py
```

### Live Interactive Demonstration

For the review and evaluation panel demonstration:

```bash
# Run on sample drone image
python demo.py --source test_images/drone.jpg

# Run on live webcam or custom video
python demo.py --source 0
python demo.py --source results/distance/formation_demo.mp4
```

### Running Phases Individually

Run the scripts from the project root in sequence:

```bash
python phase1_test.py
python phase2_position.py
python phase3_formation.py
python phase4_evaluation.py
python phase5_visualization.py
python phase6_formation_map.py
python phase7_normalized_distance.py
python phase8_formation_statistics.py
python phase9_visualization.py
python phase10_distance_estimation.py
```

Trained detector weights are located at:
`runs/detect/runs/drone/formation_v1/weights/best.pt`

## Mandatory Mini-Project Deliverables

1. **Source Code & Verification:**
   - Complete 10-phase pipeline and testing tools.
   - Master verification harness: [`verify_pipeline.py`](file:///Users/manasvidr/Documents/GitHub/Vision-Based-Pose-Estimation-/verify_pipeline.py).
   - Live demonstration runner: [`demo.py`](file:///Users/manasvidr/Documents/GitHub/Vision-Based-Pose-Estimation-/demo.py).

2. **Executive Two-Page Project Write-up:**
   - **PDF Document:** [`PROJECT_WRITEUP.pdf`](file:///Users/manasvidr/Documents/GitHub/Vision-Based-Pose-Estimation-/PROJECT_WRITEUP.pdf)
   - **Markdown Document:** [`PROJECT_WRITEUP.md`](file:///Users/manasvidr/Documents/GitHub/Vision-Based-Pose-Estimation-/PROJECT_WRITEUP.md)

3. **Presentation Slide Deck & Viva Q&A:**
   - **Interactive Browser Slide Deck:** [`presentation.html`](file:///Users/manasvidr/Documents/GitHub/Vision-Based-Pose-Estimation-/presentation.html)
   - **Slide Notes & Oral Q&A Prep:** [`PRESENTATION_SLIDES.md`](file:///Users/manasvidr/Documents/GitHub/Vision-Based-Pose-Estimation-/PRESENTATION_SLIDES.md)

## Repository Structure

- `phase1_test.py` — Initial detection testing
- `phase2_position.py` — 2D position extraction
- `phase3_formation.py` — Relative formation analysis
- `phase4_evaluation.py` — Model evaluation
- `phase5_visualization.py` — Annotated detection and formation visualization
- `phase6_formation_map.py` — Normalized formation mapping
- `phase7_normalized_distance.py` — Normalized pairwise distances
- `phase8_formation_statistics.py` — Formation statistics
- `phase9_visualization.py` — Statistical charts
- `phase10_distance_estimation.py` — Approximate distance from box width, formation status, error analysis
- `tools/manasvi_distance_viz.py` — Calibration, image/video overlay and error analysis for a real camera
- `tools/edge_case_tests.py` — Detector tests on blur, darkness, cropping and out-of-distribution images
- `results/distance/` — Phase 10 outputs (metrics, plots, demo video)

Evaluation metrics, formation CSVs, and selected Phase 9 outputs are included in the `runs` directory.

## Limitations

Phases 1-9 analyse 2D image coordinates only; their normalized distances are not metres.

Phase 10 adds an approximate distance from box width, Z = f x W / w. It assumes a drone width of 0.46 m (3DR Solo) and an assumed 60 degree horizontal field of view, because the dataset video has no camera data. The metre values are therefore illustrative, not measured, and no physical distance measurements have been taken yet. The distance error reported in `results/distance/report_numbers.txt` covers box-width error only, not error in W, f or the pinhole model. Formation status uses spacing in drone-widths, which needs no calibration.

This is not full 3D pose reconstruction. The detector is strong on images like the training video but unreliable outside it (see `tools/edge_case_tests.py`): a close-up white drone gives partial boxes, a street scene with no drone gives false detections, and heavy motion blur loses the drone. The test split comes from the same video as the training data, so test scores are optimistic.

Camera calibration, a measured reference size and validation data would be needed for physical measurements.

## Dataset and Reproducibility

The dataset used for training and evaluation is not included in this repository. Obtain it separately and update the dataset paths in the configuration as needed. A compatible Python environment and the trained model weights are also required to reproduce the results.

## Future Work

- Improve robustness on varied drone images and environments.
- Run the physical distance test below to replace the assumed camera values.
- Fine-tune on close-up and white drones and add negative images.
- Investigate camera calibration and geometric reconstruction.
- Explore 3D pose estimation with appropriate reference data.
- Validate performance on independent datasets

## Phase 10: Distance estimation and formation status (Manasvi D R)

`phase10_distance_estimation.py` estimates approximate distance from box width (Z = f x W / w, with W = 0.46 m for the 3DR Solo and an assumed 60 degree field of view), measures nearest-neighbour spacing in drone-widths, assigns a formation status (maintained / too close / too far) and draws an overlay with a top-view map. It also measures the distance error caused by box-width error against the ground-truth labels. Spacing in drone-widths needs no calibration, unlike the image-size normalization in Phase 7. Metre values depend on the assumed camera.

    python3 phase10_distance_estimation.py --positions runs/phase2_position/drone_positions.csv --images <test/images> --labels <test/labels> --out results/distance

`tools/` has a calibration script for a real camera and edge-case tests of the detector on a close-up white drone and a street scene (it produces false detections and loses a blurred drone).

### Physical distance test (to validate Phase 10)

1. Photograph one object of known width (a drone or any box) at known distances, for example 1, 2, 3 and 5 m, with the same camera and zoom.
2. Calibrate the focal length from the 2 m photo (use `--width-m` for the real width of the object in metres):

        python3 tools/manasvi_distance_viz.py calibrate --image photo_2m.jpg --distance 2.0 --width-m 0.46

3. Estimate the distance in each photo with the printed focal length:

        python3 tools/manasvi_distance_viz.py run --source photo_3m.jpg --focal <value> --width-m 0.46

4. Fill `tools/measurements_template.csv` (columns `true_m`, `estimated_m`) and run:

        python3 tools/manasvi_distance_viz.py errors --csv tools/measurements_template.csv

Note: the detector is trained on small dark drones against sky, so use an object it detects, or check the box before trusting the result.
