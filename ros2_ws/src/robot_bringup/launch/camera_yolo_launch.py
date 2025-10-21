from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Camera node
        Node(
            package='sensors_pkg',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[
                # Add any parameters here if needed
            ]
        ),

        # YOLO detection node with visualization
        Node(
            package='perception_pkg',
            executable='yolo_node',
            name='yolo_node',
            output='screen',
            parameters=[
                # Add any parameters here if needed
            ]
        ),
    ])
