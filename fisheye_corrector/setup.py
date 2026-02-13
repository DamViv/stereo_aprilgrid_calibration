from setuptools import setup
import glob
import os

package_name = 'fisheye_corrector'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/calib', ['calib/camera_0.yaml', 'calib/camera_1.yaml', 'calib/stereo_calibration.yaml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'numpy', 'opencv-python'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='ROS 2 Python node for fisheye image correction using Double Sphere model',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'fisheye_corrector_node = fisheye_corrector.fisheye_corrector_node:main',
            'fisheye_disparity_node = fisheye_corrector.fisheye_disparity_node:main',
        ],
    },
)