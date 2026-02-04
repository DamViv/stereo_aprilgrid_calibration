from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # Arguments configurables
    image_topics_arg = DeclareLaunchArgument(
        "image_topics",
        default_value="/cam_0/image_raw,/cam_1/image_raw",
        description="Comma-separated list of image topics"
    )
    output_topics_arg = DeclareLaunchArgument(
        "output_topics",
        default_value="/cam_0/apriltags,/cam_1/apriltags",
        description="Comma-separated list of output topics"
    )

    tag_family_arg = DeclareLaunchArgument(
        "tag_family",
        default_value="36h11",
        description="AprilTag family"
    )

    tag_size_arg = DeclareLaunchArgument(
        "tag_size",
        default_value="0.5",
        description="Tag size in meters"
    )

    fx_arg = DeclareLaunchArgument("fx", default_value="600.0")
    fy_arg = DeclareLaunchArgument("fy", default_value="600.0")
    px_arg = DeclareLaunchArgument("px", default_value="320.0")
    py_arg = DeclareLaunchArgument("py", default_value="240.0")

    width_arg  = DeclareLaunchArgument("width",  default_value="640")
    height_arg = DeclareLaunchArgument("height", default_value="480")

    display_arg = DeclareLaunchArgument("display", default_value="true", description="Display OpenCV window")

    # Node
    aprilgrid_node = Node(
        package="aprilgrid_detector",
        executable="aprilgrid_detector_node",
        name="aprilgrid_detector",
        output="screen",

        parameters=[{
            "image_topics": LaunchConfiguration("image_topics"),
            "output_topics": LaunchConfiguration("output_topics"),
            "tag_family":  LaunchConfiguration("tag_family"),
            "tag_size":    LaunchConfiguration("tag_size"),

            "width": LaunchConfiguration("width"),
            "height": LaunchConfiguration("height"),
            "fx": LaunchConfiguration("fx"),
            "fy": LaunchConfiguration("fy"),
            "px": LaunchConfiguration("px"),
            "py": LaunchConfiguration("py"),

            "display": LaunchConfiguration("display"),
        }]
    )

    return LaunchDescription([
        image_topics_arg,
        output_topics_arg,
        tag_family_arg,
        tag_size_arg,
        fx_arg, fy_arg, px_arg, py_arg,
        width_arg, height_arg,
        display_arg,

        aprilgrid_node
    ])

