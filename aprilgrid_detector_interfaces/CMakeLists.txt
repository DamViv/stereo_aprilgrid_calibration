cmake_minimum_required(VERSION 3.5)
project(aprilgrid_detector_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)


rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AprilTag.msg"
  "msg/AprilTagArray.msg"
  "msg/CalibrationStats.msg"
  "srv/StartCollection.srv"
  "srv/StopCollection.srv"
  "srv/ClearFrames.srv"
  "srv/CalibrateCamera.srv"
  "srv/CalibrateStereo.srv"
  DEPENDENCIES std_msgs
)

ament_package()
