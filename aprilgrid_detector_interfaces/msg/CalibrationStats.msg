# Statistics about collected calibration frames

std_msgs/Header header

# Frame counts
uint32 cam0_frame_count
uint32 cam1_frame_count
uint32 stereo_pair_count

# Quality metrics (0.0 to 1.0)
float32 cam0_coverage      # Image area coverage
float32 cam1_coverage
float32 cam0_diversity     # Pose diversity metric
float32 cam1_diversity

# Readiness flags
bool cam0_ready           # Enough frames to calibrate cam0
bool cam1_ready           # Enough frames to calibrate cam1
bool stereo_ready         # Ready for stereo calibration (both cams calibrated + pairs)

# Collection state
bool is_collecting
string collection_mode    # "IDLE", "MONO_CAM0", "MONO_CAM1", "STEREO", "BOTH"
