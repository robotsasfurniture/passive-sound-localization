from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("passive_sound_localization"),
        "localization_params.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="passive_sound_localization",
                executable="localization_node",
                name="localization_node",
                output="screen",
                parameters=[
                    config,
                ],
            )
        ]
    )
