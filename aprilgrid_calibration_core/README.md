# AprilGrid Calibration Core

Core calibration logic with frame collection and camera calibration services.

## Overview

This package provides:
- Frame collection with intelligent filtering
- Monocular camera calibration (Double Sphere model)
- Stereo camera calibration
- ROS2 services for GUI control

## Nodes

### calibration_manager_node

Main node that manages the calibration pipeline.

**Subscribed Topics:**
- `/cam_0/apriltags` (AprilTagArray)
- `/cam_1/apriltags` (AprilTagArray)

**Published Topics:**
- `/calibration/stats` (CalibrationStats)

**Services:**
- `/calibration/start_collection` (StartCollection)
- `/calibration/stop_collection` (StopCollection)
- `/calibration/clear_frames` (ClearFrames)
- `/calibration/calibrate_camera` (CalibrateCamera)
- `/calibration/calibrate_stereo` (CalibrateStereo)

## Usage

```bash
# Launch everything
ros2 launch aprilgrid_calibration_core calibration.launch.py

# Or launch individually
ros2 run aprilgrid_calibration_core calibration_manager_node
```

## Configuration

Edit `config/calibration_params.yaml` to adjust:
- Image dimensions
- AprilGrid geometry
- Collection thresholds
- Calibration parameters

## Dependencies

- ROS2 (Jazzy)
- OpenCV 4.x
- Eigen3
- aprilgrid_detector_interfaces
