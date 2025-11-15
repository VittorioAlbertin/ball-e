#!/usr/bin/env python3
"""
Person Enrollment Launch File

DESCRIPTION:
    This launch file starts the components needed for interactive person enrollment:
    1. Camera Node - For face capture
    2. Microphone Node - For voice capture
    3. Face Detector - Face detection service
    4. Face Recognizer - Face embedding generation service
    5. Voice Recognizer - Voice embedding generation service
    6. People Database - Database service for storing identities

    After launching, run the enrollment CLI:
        ros2 run interaction_pkg enrollment_cli

WORKFLOW:
    User runs enrollment_cli which:
    1. Prompts for person's name
    2. Captures 5 face poses (front, left, right, up, down)
    3. Captures 3 voice samples
    4. Computes quality-weighted embeddings
    5. Saves to database

USAGE:
    ros2 launch robot_bringup enrollment_launch.py

    Then in another terminal:
        ros2 run interaction_pkg enrollment_cli

    Optional arguments:
    - camera_index:=0 (default: 0)
    - use_gpu:=true (default: true)
    - enable_voice:=true (default: true)

AUTHOR: Ball-e Team
DATE: 2025-11-15
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    camera_index_arg = DeclareLaunchArgument(
        'camera_index',
        default_value='0',
        description='Camera device index'
    )

    use_gpu_arg = DeclareLaunchArgument(
        'use_gpu',
        default_value='true',
        description='Use GPU acceleration for face recognition'
    )

    enable_voice_arg = DeclareLaunchArgument(
        'enable_voice',
        default_value='true',
        description='Enable voice enrollment'
    )

    # Get launch configurations
    camera_index = LaunchConfiguration('camera_index')
    use_gpu = LaunchConfiguration('use_gpu')
    enable_voice = LaunchConfiguration('enable_voice')

    # ===== VISUAL PIPELINE (for face enrollment) =====
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
            'publish_fps': 30.0,
        }]
    )

    face_detector = Node(
        package='perception_pkg',
        executable='face_detector_node',
        name='face_detector',
        output='screen',
        parameters=[{
            'confidence_threshold': 0.6,
            'nms_threshold': 0.3,
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

    # ===== AUDIO PIPELINE (for voice enrollment) =====
    microphone_node = Node(
        package='sensors_pkg',
        executable='microphone_node',
        name='microphone_node',
        output='screen',
        condition=IfCondition(enable_voice),
        parameters=[{
            'sample_rate': 16000,
            'chunk_duration_ms': 200,
            'vad_threshold': 0.01,
            'speech_min_duration': 0.3,
            'speech_max_silence': 0.5,
            'publish_raw_audio': False,
        }]
    )

    voice_recognizer_node = Node(
        package='perception_pkg',
        executable='voice_recognizer_node',
        name='voice_recognizer',
        output='screen',
        condition=IfCondition(enable_voice),
        parameters=[{
            'auto_process_speech': False,  # CLI will call service directly
        }]
    )

    # ===== DATABASE SERVICE =====
    people_database = Node(
        package='interaction_pkg',
        executable='people_database_node',
        name='people_database_node',
        output='screen',
        parameters=[{
            'db_path': '/ball-e/ros2_ws/robot_data/people.db',
        }]
    )

    # Info message
    startup_info = LogInfo(
        msg="\n" + "="*60 + "\n" +
            "  ENROLLMENT MODE READY\n" +
            "  Run enrollment CLI: ros2 run interaction_pkg enrollment_cli\n" +
            "="*60
    )

    # Return launch description
    return LaunchDescription([
        # Launch arguments
        camera_index_arg,
        use_gpu_arg,
        enable_voice_arg,

        # Startup info
        startup_info,

        # Visual pipeline
        camera_node,
        face_detector,
        face_recognizer,

        # Audio pipeline (conditional)
        microphone_node,
        voice_recognizer_node,

        # Database service
        people_database,
    ])
