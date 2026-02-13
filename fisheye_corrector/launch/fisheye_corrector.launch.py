from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Chemin absolu vers le fichier de calibration
    pkg_dir = os.path.dirname(os.path.realpath(__file__))  # launch/
    pkg_dir = os.path.abspath(os.path.join(pkg_dir, '..'))  # package root
    calib_file0 = os.path.join(pkg_dir, 'calib', 'camera_0.yaml')
    calib_file1 = os.path.join(pkg_dir, 'calib', 'camera_1.yaml')
    calib_stereo = os.path.join(pkg_dir, 'calib', 'stereo_calibration.yaml')

    return LaunchDescription([
        Node(
            package='fisheye_corrector',
            executable='fisheye_corrector_node',
            name='fisheye_corrector_node',
            output='screen',
            parameters=[
                {'image_topic': '/cam_0/image_raw'},
                {'corrected_topic': '/cam_0/image_corrected'},
                {'calib_file': calib_file0},
                {'output_size': [640, 480]},
                {'method': 'perspective'}
            ]
        ),
        Node(
            package='fisheye_corrector',
            executable='fisheye_corrector_node',
            name='fisheye_corrector_node',
            output='screen',
            parameters=[
                {'image_topic': '/cam_1/image_raw'},
                {'corrected_topic': '/cam_1/image_corrected'},
                {'calib_file': calib_file1},
                {'output_size': [640, 480]},
                {'method': 'perspective'}
            ]
        ),
        Node(
            package='fisheye_corrector',
            executable='fisheye_disparity_node',
            name='fisheye_disparity_node',
            output='screen',
            parameters=[
                {'image_topic_left': '/cam_0/image_raw'},
                {'image_topic_right': '/cam_1/image_raw'},
                {'disparity_topic': '/stereo/disparity_image'},
                {'calib_file': calib_stereo},
                {'output_size': [640, 480]},
                {'method': 'perspective'}
            ]
        )
    ])