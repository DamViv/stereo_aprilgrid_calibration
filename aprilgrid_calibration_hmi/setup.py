from setuptools import setup

package_name = 'aprilgrid_calibration_hmi'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'PySide6'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='GUI pour la calibration des caméras AprilGrid',
    license='MIT',
    entry_points={
        'console_scripts': [
            'calibration_gui = aprilgrid_calibration_hmi.calibration_gui:main'
        ],
    },
)
