from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Launch file for person tracking with ByteTrack.

    This launch file starts the person tracker node with configurable parameters
    for tracking behavior. It requires YOLO detections to be published on /yolo/detections.

    Parameters:
        max_age: Maximum frames to keep a track alive without detection (default: 30)
        min_hits: Minimum consecutive detections before confirming track (default: 3)
        iou_threshold: IoU threshold for matching detections to tracks (default: 0.3)
        high_conf_threshold: Confidence threshold for high-quality detections (default: 0.6)
        low_conf_threshold: Minimum confidence for low-quality detections (default: 0.1)
    """

    # Declare launch arguments
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

    high_conf_threshold_arg = DeclareLaunchArgument(
        'high_conf_threshold',
        default_value='0.6',
        description='Confidence threshold for high-quality detections (ByteTrack stage 1)'
    )

    low_conf_threshold_arg = DeclareLaunchArgument(
        'low_conf_threshold',
        default_value='0.1',
        description='Minimum confidence for low-quality detections (ByteTrack stage 2)'
    )

    # Person tracker node
    person_tracker_node = Node(
        package='perception_pkg',
        executable='person_tracker',
        name='person_tracker',
        output='screen',
        parameters=[{
            'max_age': LaunchConfiguration('max_age'),
            'min_hits': LaunchConfiguration('min_hits'),
            'iou_threshold': LaunchConfiguration('iou_threshold'),
            'high_conf_threshold': LaunchConfiguration('high_conf_threshold'),
            'low_conf_threshold': LaunchConfiguration('low_conf_threshold'),
        }]
    )

    return LaunchDescription([
        max_age_arg,
        min_hits_arg,
        iou_threshold_arg,
        high_conf_threshold_arg,
        low_conf_threshold_arg,
        person_tracker_node,
    ])
