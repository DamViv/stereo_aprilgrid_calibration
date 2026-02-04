from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='aprilgrid_calibration_hmi',
            executable='calibration_gui',
            name='calibration_gui',
        )
    ])
