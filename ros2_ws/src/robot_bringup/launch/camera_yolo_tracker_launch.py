from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Complete perception pipeline: Camera → YOLO → Person Tracker

    This launch file starts the complete perception pipeline for Ball-e:
    1. Camera node: Captures video from camera
    2. YOLO node: Detects objects including people
    3. Person tracker: Tracks people with persistent IDs using ByteTrack
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

        # Launch arguments for person tracker
        max_age_arg,
        min_hits_arg,
        iou_threshold_arg,

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
    ])
