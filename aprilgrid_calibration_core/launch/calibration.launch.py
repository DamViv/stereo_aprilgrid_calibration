#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression

from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directories
    core_pkg_dir = get_package_share_directory('aprilgrid_calibration_core')    
    ds_pkg_dir = get_package_share_directory('double_sphere_corrector')

    # Config files
    calibration_config = os.path.join(core_pkg_dir, 'config', 'calibration_params.yaml')       
    stereo_calibration_file = os.path.join(ds_pkg_dir, 'calib', 'stereo_calibration.yaml') 

    # AprilGrid Detector for Cam0
    detector = Node(
        package='aprilgrid_detector',
        executable='aprilgrid_detector_node',
        name='aprilgrid_detector_cam0',
        namespace='cam_0',
        parameters=[{
            'image_topic': '/cam_0/image_raw,/cam_1/image_raw',
            'output_topic': '/cam_0/apriltags,/cam_1/apriltags',
            'tag_family': '36h11',
            'tag_size': 0.5,
            'display': False,
        }],
        output='screen'
    )    
    
    # Calibration Manager
    calibration_manager = Node(
        package='aprilgrid_calibration_core',
        executable='calibration_manager_node',
        name='calibration_manager',
        parameters=[calibration_config],
        output='screen'
    )
    
    # Calibration GUI
    calibration_gui = Node(
        package='aprilgrid_calibration_hmi',
        executable='calibration_gui',
        name='calibration_gui',
        output='screen'
    )
    
    # Calibration GUI
    stereo_3D_reconstruction = Node(
        package='double_sphere_corrector',
        executable='ds_corrector',
        name='ds_corrector',
        parameters=[{
            'img_left_topic': '/cam_0/image_raw',
            'img_right_topic': '/cam_1/image_raw',
            'output_topic': '/cam_0/apriltags,/cam_1/apriltags',
            'stereo_config_file': stereo_calibration_file,
        }],
        output='screen'
    )

    return LaunchDescription([
        detector,        
        calibration_manager,
        calibration_gui,
        stereo_3D_reconstruction,
    ])
