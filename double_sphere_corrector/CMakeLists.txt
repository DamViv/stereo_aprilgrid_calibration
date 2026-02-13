cmake_minimum_required(VERSION 3.8)
project(double_sphere_corrector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rcutils REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)


# Include directories
include_directories(
  ${rclcpp_INCLUDE_DIRS} 
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS} 
  ${YAML_CPP_INCLUDE_DIR} 
  ${Boost_SYSTEM_LIBRARY} 
  ${cv_bridge_INCLUDE_DIR}
)

# -------------------------------------------------------------------
# Library
# -------------------------------------------------------------------

file(GLOB_RECURSE LIB_SOURCES src/*.cpp include/*.cpp)
file(GLOB_RECURSE LIB_HEADERS include/*.h include/*.hpp)
add_library(${PROJECT_NAME} SHARED ${LIB_SOURCES} ${LIB_HEADERS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES} ${cv_bridge_LIBRARIES} ament_index_cpp::ament_index_cpp)

install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  RUNTIME DESTINATION bin/${PROJECT_NAME}
)


# -------------------------------------------------------------------
# Executables generation
# -------------------------------------------------------------------

add_executable(ds_corrector
  src/double_sphere_corrector_node.cpp
  src/sphericRectification.cpp
  src/denseStereo.cpp
)
ament_target_dependencies(${PROJECT_NAME} 
  rclcpp 
  sensor_msgs 
  image_transport 
  cv_bridge 
  pcl_conversions 
  ament_index_cpp
  yaml-cpp 
)
target_link_libraries(ds_corrector ${PROJECT_NAME})


# Install
install(TARGETS
  ds_corrector 
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY calib
  DESTINATION share/${PROJECT_NAME}/
)  

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)  
ament_package()
