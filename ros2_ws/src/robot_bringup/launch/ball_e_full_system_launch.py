from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Complete Ball-e tracking and identification system:

    Camera → YOLO → Person Tracker → Person State Manager
                                            ↓
                    Identification Coordinator → Face Recognition (conditional)
                                            ↓
                                    Visualization Node
                                            ↓
                                    People Database

    This is the full optimized pipeline with:
    - Persistent person tracking (ByteTrack)
    - Centralized state management
    - Smart identification coordination
    - On-demand face recognition
    - Rich visualization with track IDs and identities
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

    # Coordinator parameters
    max_requests_per_second_arg = DeclareLaunchArgument(
        'max_requests_per_second',
        default_value='2.0',
        description='Maximum identification requests per second'
    )

    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Re-identify below this confidence'
    )

    # Face recognition parameters
    recognition_threshold_arg = DeclareLaunchArgument(
        'recognition_threshold',
        default_value='0.6',
        description='Minimum cosine similarity for face matching'
    )

    reidentification_interval_arg = DeclareLaunchArgument(
        'reidentification_interval',
        default_value='60.0',
        description='Seconds before re-identifying known persons'
    )

    return LaunchDescription([
        # Launch arguments
        max_age_arg,
        min_hits_arg,
        iou_threshold_arg,
        cleanup_timeout_arg,
        max_requests_per_second_arg,
        confidence_threshold_arg,
        recognition_threshold_arg,
        reidentification_interval_arg,

        # ========== SENSORS ==========
        Node(
            package='sensors_pkg',
            executable='camera_node',
            name='camera_node',
            output='screen',
        ),

        # ========== DETECTION ==========
        Node(
            package='perception_pkg',
            executable='yolo_node',
            name='yolo_node',
            output='screen',
        ),

        # ========== TRACKING ==========
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

        # ========== STATE MANAGEMENT ==========
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

        # ========== IDENTIFICATION ==========
        Node(
            package='perception_pkg',
            executable='identification_coordinator',
            name='identification_coordinator',
            output='screen',
            parameters=[{
                'max_requests_per_second': LaunchConfiguration('max_requests_per_second'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'recheck_interval': LaunchConfiguration('reidentification_interval'),
                'new_track_delay': 1.0,
                'enable_auto_identification': True,
            }]
        ),

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

        # ========== DATABASE ==========
        Node(
            package='interaction_pkg',
            executable='people_database_node',
            name='people_database_node',
            output='screen',
        ),

        # ========== VISUALIZATION ==========
        Node(
            package='perception_pkg',
            executable='visualization_node',
            name='visualization_node',
            output='screen',
            parameters=[{
                'show_track_id': True,
                'show_identity': True,
                'show_confidence': True,
                'show_status': True,
                'box_thickness': 2,
                'font_scale': 0.6,
            }]
        ),
    ])
