#!/usr/bin/env python3
"""
Audio Pipeline Launch File

DESCRIPTION:
    This launch file starts the audio capture and processing pipeline:
    1. Microphone Node - Audio capture with Voice Activity Detection
    2. Voice Recognizer Node - Speaker embedding generation using Resemblyzer
    3. Speech-to-Text Node - Transcription using Whisper

DATA FLOW:
    Microphone → Speech Segments → Voice Recognizer → Speaker Embeddings
                                 → Speech-to-Text → Transcriptions

USAGE:
    ros2 launch robot_bringup audio_pipeline_launch.py

    Optional arguments:
    - sample_rate:=16000 (default: 16000)
    - vad_threshold:=0.01 (default: 0.01)
    - whisper_model:='base' (default: 'base')
    - enable_stt:=true (default: true)

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
    sample_rate_arg = DeclareLaunchArgument(
        'sample_rate',
        default_value='16000',
        description='Audio sample rate in Hz'
    )

    whisper_model_arg = DeclareLaunchArgument(
        'whisper_model',
        default_value='base',
        description='Whisper model size (tiny, base, small, medium)'
    )

    enable_stt_arg = DeclareLaunchArgument(
        'enable_stt',
        default_value='true',
        description='Enable Speech-to-Text transcription'
    )

    # Get launch configurations
    sample_rate = LaunchConfiguration('sample_rate')
    whisper_model = LaunchConfiguration('whisper_model')
    enable_stt = LaunchConfiguration('enable_stt')

    # Node definitions
    microphone_node = Node(
        package='sensors_pkg',
        executable='microphone_node',
        name='microphone_node',
        output='screen',
        parameters=[{
            'sample_rate': sample_rate,
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

    # Return launch description with all nodes
    return LaunchDescription([
        # Launch arguments
        sample_rate_arg,
        whisper_model_arg,
        enable_stt_arg,

        # Audio capture
        microphone_node,

        # Voice processing
        voice_recognizer_node,
        speech_to_text_node,
    ])
