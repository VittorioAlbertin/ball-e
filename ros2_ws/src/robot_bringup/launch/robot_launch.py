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
                # {'param_name': 'value'}
            ]
        ),
        
        # YOLO detection node
        Node(
            package='perception_pkg',
            executable='yolo_node',
            name='yolo_node',
            output='screen',
            parameters=[
                # Add any parameters here if needed
            ]
        ),
        
        # Face detection node
        Node(
            package='perception_pkg',
            executable='face_detection_node',
            name='face_detection_node',
            output='screen',
            parameters=[
                # Add any parameters here if needed
            ]
        ),
        
        # People database node
        Node(
            package='interaction_pkg',
            executable='people_database_node',
            name='people_database_node',
            output='screen',
            parameters=[
                # Add any parameters here if needed
            ]
        ),
    ])