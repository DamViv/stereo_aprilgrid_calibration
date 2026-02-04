cmake_minimum_required(VERSION 3.8)
project(aprilgrid_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(aprilgrid_detector_interfaces REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

include_directories(
  include
  ${rclcpp_INCLUDE_DIRS} 
  ${OpenCV_INCLUDE_DIRS}  
  ${EIGEN3_INCLUDE_DIRS}
)

###########
# LIBRARY #
###########
file(GLOB_RECURSE LIB_SOURCES src/*.cpp src/*.cc)
file(GLOB_RECURSE LIB_HEADERS include/*.h include/*.hpp)
add_library(${PROJECT_NAME} SHARED ${LIB_SOURCES} ${LIB_HEADERS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES} ${cv_bridge_LIBRARIES} ament_index_cpp::ament_index_cpp)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  sensor_msgs
  image_transport
  cv_bridge
  aprilgrid_detector_interfaces
)

install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  RUNTIME DESTINATION bin/${PROJECT_NAME}
)

############
# Executables generation #
############
add_executable(aprilgrid_detector_node src/aprilgrid_detector_node.cpp)

target_link_libraries(aprilgrid_detector_node
  ${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
)

install(TARGETS
  aprilgrid_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
