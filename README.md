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

Evaluation metrics, formation CSVs, and selected Phase 9 outputs are included in the `runs` directory.

## Limitations

The current implementation analyzes 2D image coordinates. Normalized distances are not physical distances in metres. Full 3D pose reconstruction and real-world distance estimation have not been implemented.

Camera calibration, suitable geometric references, and validation data would be needed for further development toward physical 3D measurements.

## Dataset and Reproducibility

The dataset used for training and evaluation is not included in this repository. Obtain it separately and update the dataset paths in the configuration as needed. A compatible Python environment and the trained model weights are also required to reproduce the results.

## Future Work

- Improve robustness on varied drone images and environments.
- Investigate camera calibration and geometric reconstruction.
- Explore 3D pose estimation with appropriate reference data.
- Validate performance on independent datasets.