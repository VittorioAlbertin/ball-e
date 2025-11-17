#!/usr/bin/env python3
"""
Multi-Modal Identification Pipeline Launch File

DESCRIPTION:
    This launch file starts the complete multi-modal person identification system
    with Bayesian fusion of face and voice recognition:

    Visual Pipeline:
    1. Camera Node - Dual-stream synchronized publishing
    2. YOLO Node - Person detection
    3. Person Tracker - ByteTrack multi-object tracking
    4. Face Detector - Face detection in person ROIs
    5. Face Recognizer - Face embedding generation and matching

    Audio Pipeline:
    6. Microphone Node - Audio capture with VAD
    7. Voice Recognizer Node - Speaker embedding generation
    8. Speech-to-Text Node - Transcription (optional)

    Fusion & Management:
    9. Person State Manager - World state with Bayesian identity fusion
    10. People Database - Database service for identity management
    11. Visualization - Display identification results

DATA FLOW:
    Camera → YOLO → Tracker ────────┐
                                     ↓
    Microphone → Voice Recognizer → Person State Manager → Database
                                     ↑
    Face Detector ← Face Recognizer ─┘

    Identity = Bayesian Fusion(Face Scores, Voice Scores)

USAGE:
    ros2 launch robot_bringup multimodal_identification_launch.py

    Optional arguments:
    - camera_index:=0 (default: 0)
    - use_gpu:=true (default: true)
    - enable_voice:=true (default: true)
    - enable_stt:=true (default: true)
    - whisper_model:='base' (default: 'base')
    - identity_confidence_threshold:=0.6 (default: 0.6)

AUTHOR: Ball-e Team
DATE: 2025-11-15
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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

    publish_fps_arg = DeclareLaunchArgument(
        'publish_fps',
        default_value='30.0',
        description='Frame rate for camera streams'
    )

    use_gpu_arg = DeclareLaunchArgument(
        'use_gpu',
        default_value='true',
        description='Use GPU acceleration for face recognition'
    )

    enable_voice_arg = DeclareLaunchArgument(
        'enable_voice',
        default_value='true',
        description='Enable voice recognition pipeline'
    )

    enable_stt_arg = DeclareLaunchArgument(
        'enable_stt',
        default_value='true',
        description='Enable Speech-to-Text transcription'
    )

    whisper_model_arg = DeclareLaunchArgument(
        'whisper_model',
        default_value='base',
        description='Whisper model size (tiny, base, small, medium)'
    )

    identity_confidence_threshold_arg = DeclareLaunchArgument(
        'identity_confidence_threshold',
        default_value='0.6',
        description='Minimum Bayesian confidence for identity assignment'
    )

    # Get launch configurations
    camera_index = LaunchConfiguration('camera_index')
    publish_fps = LaunchConfiguration('publish_fps')
    use_gpu = LaunchConfiguration('use_gpu')
    enable_voice = LaunchConfiguration('enable_voice')
    enable_stt = LaunchConfiguration('enable_stt')
    whisper_model = LaunchConfiguration('whisper_model')
    identity_confidence_threshold = LaunchConfiguration('identity_confidence_threshold')

    # ===== VISUAL PIPELINE =====
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

    # ===== AUDIO PIPELINE =====
    microphone_node = Node(
        package='sensors_pkg',
        executable='microphone_node',
        name='microphone_node',
        output='screen',
        condition=IfCondition(enable_voice),
        parameters=[{
            'sample_rate': 16000,
            'chunk_duration_ms': 200,
            'vad_threshold_start': 0.025,
            'vad_threshold_end': 0.015,
            'speech_min_duration': 0.5,
            'speech_max_silence': 1.5,
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
            'auto_process_speech': True,
        }]
    )

    speech_to_text_node = Node(
        package='perception_pkg',
        executable='speech_to_text_node',
        name='speech_to_text',
        output='screen',
        condition=IfCondition(enable_stt),
        parameters=[{
            'model_size': whisper_model,
            'language': 'en',
            'auto_detect_language': False,
        }]
    )

    # ===== FUSION & MANAGEMENT =====
    person_state_manager = Node(
        package='perception_pkg',
        executable='person_state_manager_node',
        name='person_state_manager',
        output='screen',
        parameters=[{
            'identity_confidence_threshold': identity_confidence_threshold,
            'known_person_reidentify_interval': 60.0,
            'unknown_person_reidentify_interval': 15.0,
            'max_identification_attempts': 5,
            'enable_voice_recognition': enable_voice,
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
        enable_voice_arg,
        enable_stt_arg,
        whisper_model_arg,
        identity_confidence_threshold_arg,

        # Visual pipeline
        camera_node,
        yolo_node,
        person_tracker,
        face_detector,
        face_recognizer,

        # Audio pipeline (conditional)
        microphone_node,
        voice_recognizer_node,
        speech_to_text_node,

        # Fusion & management
        person_state_manager,
        people_database,
        identification_visualization,
    ])
