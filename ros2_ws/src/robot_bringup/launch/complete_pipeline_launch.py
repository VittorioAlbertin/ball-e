from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Complete Ball-e perception and recognition pipeline:
    Camera → YOLO → Person Tracker → State Manager → Conditional Face Recognition

    This is the full optimized pipeline with on-demand face recognition,
    addressing the original ~1 FPS bottleneck.
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

    # Face recognition parameters
    recognition_threshold_arg = DeclareLaunchArgument(
        'recognition_threshold',
        default_value='0.6',
        description='Minimum cosine similarity for face matching'
    )

    reidentification_interval_arg = DeclareLaunchArgument(
        'reidentification_interval',
        default_value='30.0',
        description='Seconds before re-identifying known persons'
    )

    return LaunchDescription([
        # Launch arguments
        max_age_arg,
        min_hits_arg,
        iou_threshold_arg,
        cleanup_timeout_arg,
        recognition_threshold_arg,
        reidentification_interval_arg,

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

        # Conditional face recognition node
        Node(
            package='perception_pkg',
            executable='face_recognition_conditional',
            name='face_recognition_conditional',
            output='screen',
            parameters=[{
                'recognition_threshold': LaunchConfiguration('recognition_threshold'),
                'min_face_size': 20,
                'max_face_size': 400,
                'frame_cache_size': 10,
                'reidentification_interval': LaunchConfiguration('reidentification_interval'),
                'auto_identify_new_tracks': True,
            }]
        ),

        # People database service node
        Node(
            package='interaction_pkg',
            executable='people_database_node',
            name='people_database_node',
            output='screen',
        ),
    ])
