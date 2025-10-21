from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Complete perception pipeline for Ball-e:
    Camera → YOLO → Person Tracker → Person State Manager

    This launch file starts the complete person tracking and state management
    pipeline, creating the foundation for conditional face recognition.
    """

    # Person tracker parameters
    max_age_arg = DeclareLaunchArgument(
        'max_age',
        default_value='30',
        description='Maximum frames to keep track alive without detection'
    )

    min_hits_arg = DeclareLaunchArgument(
        'min_hits',
        default_value='3',
        description='Minimum consecutive detections before confirming track'
    )

    iou_threshold_arg = DeclareLaunchArgument(
        'iou_threshold',
        default_value='0.3',
        description='IoU threshold for matching detections to tracks'
    )

    # State manager parameters
    cleanup_timeout_arg = DeclareLaunchArgument(
        'cleanup_timeout',
        default_value='5.0',
        description='Seconds before removing inactive persons'
    )

    return LaunchDescription([
        # Camera node
        Node(
            package='sensors_pkg',
            executable='camera_node',
            name='camera_node',
            output='screen',
        ),

        # YOLO detection node
        Node(
            package='perception_pkg',
            executable='yolo_node',
            name='yolo_node',
            output='screen',
        ),

        # Launch arguments
        max_age_arg,
        min_hits_arg,
        iou_threshold_arg,
        cleanup_timeout_arg,

        # Person tracker node
        Node(
            package='perception_pkg',
            executable='person_tracker',
            name='person_tracker',
            output='screen',
            parameters=[{
                'max_age': LaunchConfiguration('max_age'),
                'min_hits': LaunchConfiguration('min_hits'),
                'iou_threshold': LaunchConfiguration('iou_threshold'),
                'high_conf_threshold': 0.6,
                'low_conf_threshold': 0.1,
            }]
        ),

        # Person state manager node
        Node(
            package='perception_pkg',
            executable='person_state_manager',
            name='person_state_manager',
            output='screen',
            parameters=[{
                'cleanup_timeout': LaunchConfiguration('cleanup_timeout'),
                'publish_rate': 10.0,
            }]
        ),
    ])
