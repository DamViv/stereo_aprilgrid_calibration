cmake_minimum_required(VERSION 3.8)
project(aprilgrid_calibration_core)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(aprilgrid_detector_interfaces REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)


# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

# Library: calibration core (frame collector, calibrators, etc.)
add_library(${PROJECT_NAME}_lib
  src/frame_collector.cpp  
  src/mono_calibrator_ds.cpp
  src/stereo_calibrator_ds.cpp
  src/projections_double_sphere.cpp
)

ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  aprilgrid_detector_interfaces
  OpenCV
  yaml-cpp  
)

target_link_libraries(${PROJECT_NAME}_lib
  ${OpenCV_LIBRARIES}
  Eigen3::Eigen
)

# Executable: calibration manager node
add_executable(calibration_manager_node
  src/calibration_manager_node.cpp
)

ament_target_dependencies(calibration_manager_node
  rclcpp
  sensor_msgs
  std_msgs
  std_srvs
  cv_bridge
  aprilgrid_detector_interfaces
  OpenCV
  yaml-cpp  
)

target_link_libraries(calibration_manager_node
  ${PROJECT_NAME}_lib
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
  Eigen3::Eigen
  yaml-cpp
)

# Install
install(TARGETS
  ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  calibration_manager_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include
)

install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
