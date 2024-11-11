from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="movement_library",
                executable="movement_node",
                name="movement_node",
                output="screen",
            )
        ]
    )
