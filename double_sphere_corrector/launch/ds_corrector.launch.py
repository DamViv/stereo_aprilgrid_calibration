#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression

from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directories
    ds_pkg_dir = get_package_share_directory('double_sphere_corrector')

    # Config files
    stereo_calibration_file = os.path.join(ds_pkg_dir, 'calib', 'stereo_calibration.yaml') 
    
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
        stereo_3D_reconstruction,
    ])
