#!/usr/bin/env python3
"""
Identification Pipeline Launch File

DESCRIPTION:
    This launch file starts the complete person identification pipeline including:
    1. Camera Node - Dual-stream synchronized publishing (high-res + low-res)
    2. YOLO Node - Person detection on low-res stream
    3. Person Tracker - ByteTrack multi-object tracking
    4. Person State Manager - World state orchestrator (main logic)
    5. Face Detector - Face detection in person ROIs
    6. Face Recognizer - Face embedding generation and matching
    7. People Database - Database service for identity management

DATA FLOW:
    Camera → YOLO → Tracker → StateManager → FaceDetector → FaceRecognizer → Database
                                    ↑                                           ↓
                                    └──────────── Identity Updates ─────────────┘

USAGE:
    ros2 launch robot_bringup identification_pipeline_launch.py

    Optional arguments:
    - camera_index:=2 (default: 2)
    - publish_fps:=30.0 (default: 30.0)
    - use_gpu:=true (default: true)
    - recognition_threshold:=0.6 (default: 0.6)

AUTHOR: Vittorio Albertin
DATE: 2025-10-29
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    camera_index_arg = DeclareLaunchArgument(
        'camera_index',
        default_value='0',
        description='Camera device index (0=built-in, 2=USB)'
    )

    publish_fps_arg = DeclareLaunchArgument(
        'publish_fps',
        default_value='30.0',
        description='Frame rate for both camera streams'
    )

    use_gpu_arg = DeclareLaunchArgument(
        'use_gpu',
        default_value='true',
        description='Use GPU acceleration for face recognition'
    )

    recognition_threshold_arg = DeclareLaunchArgument(
        'recognition_threshold',
        default_value='0.6',
        description='Minimum cosine similarity for face recognition match'
    )

    # Get launch configurations
    camera_index = LaunchConfiguration('camera_index')
    publish_fps = LaunchConfiguration('publish_fps')
    use_gpu = LaunchConfiguration('use_gpu')
    recognition_threshold = LaunchConfiguration('recognition_threshold')

    # Node definitions
    camera_node = Node(
        package='sensors_pkg',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            'camera_index': camera_index,
            'fps': 30.0,
            'width': 1920,
            'height': 1080,
            'low_res_width': 640,
            'low_res_height': 360,
            'publish_fps': publish_fps,
        }]
    )

    yolo_node = Node(
        package='perception_pkg',
        executable='yolo_node',
        name='yolo_node',
        output='screen',
        parameters=[{
            'confidence_threshold': 0.3,
            'model_path': 'models/yolov5n.pt',
        }]
    )

    person_tracker = Node(
        package='perception_pkg',
        executable='person_tracker',
        name='person_tracker',
        output='screen',
        parameters=[{
            'max_age': 30,
            'min_hits': 20,
            'iou_threshold': 0.3,
            'high_conf_threshold': 0.6,
            'low_conf_threshold': 0.1,
        }]
    )

    person_state_manager = Node(
        package='perception_pkg',
        executable='person_state_manager_node',
        name='person_state_manager',
        output='screen',
        parameters=[{
            'reidentification_confidence_threshold': 0.4,
            'known_person_reidentify_interval': 60.0,
            'unknown_person_reidentify_interval': 15.0,
            'max_identification_attempts': 5,
        }]
    )

    face_detector = Node(
        package='perception_pkg',
        executable='face_detector_node',
        name='face_detector',
        output='screen',
        parameters=[{
            'confidence_threshold': 0.6,  # YuNet has well-calibrated scores
            'nms_threshold': 0.3,  # Standard NMS threshold for YuNet
        }]
    )

    face_recognizer = Node(
        package='perception_pkg',
        executable='face_recognizer_node',
        name='face_recognizer',
        output='screen',
        parameters=[{
            'use_gpu': use_gpu,
            'low_res_width': 640,
            'low_res_height': 360,
            'high_res_width': 1920,
            'high_res_height': 1080,
        }]
    )

    people_database = Node(
        package='interaction_pkg',
        executable='people_database_node',
        name='people_database_node',
        output='screen',
        parameters=[{
            'db_path': '/ball-e/ros2_ws/robot_data/people.db',
        }]
    )

    identification_visualization = Node(
        package='perception_pkg',
        executable='identification_visualization_node',
        name='identification_visualization',
        output='screen',
        parameters=[{
            'box_thickness': 2,
            'font_scale': 0.6,
            'show_confidence': True,
            'show_track_id': True,
        }]
    )

    # Return launch description with all nodes
    return LaunchDescription([
        # Launch arguments
        camera_index_arg,
        publish_fps_arg,
        use_gpu_arg,
        recognition_threshold_arg,

        # Sensor nodes
        camera_node,

        # Perception nodes
        yolo_node,
        person_tracker,

        # Identification pipeline nodes
        person_state_manager,
        face_detector,
        face_recognizer,

        # Visualization
        identification_visualization,

        # Database service
        people_database,
    ])
