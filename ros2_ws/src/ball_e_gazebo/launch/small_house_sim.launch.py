#!/usr/bin/env python3
"""
Launch file for Ball-e small house simulation with Gazebo Classic.

This launch file starts:
1. Gazebo Classic with the small_house world
2. Spawns the Ball-e robot
3. Starts robot_state_publisher
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import xacro


def generate_launch_description():
    # Get package directories
    ball_e_description_dir = get_package_share_directory('ball_e_description')
    ball_e_gazebo_dir = get_package_share_directory('ball_e_gazebo')
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')

    # Paths
    urdf_file = os.path.join(ball_e_description_dir, 'urdf', 'ball_e.urdf.xacro')
    world_file = os.path.join(ball_e_gazebo_dir, 'worlds', 'small_house.world')

    # Process xacro to get URDF
    robot_description = xacro.process_file(urdf_file).toxml()

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    # Set GAZEBO_MODEL_PATH to include small_house models
    # Point to ball_e_gazebo directory (parent of models/) so Gazebo can resolve "models/model_name" paths
    set_gazebo_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        ball_e_gazebo_dir
    )

    # Gazebo Classic launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_file,
            'verbose': 'true'
        }.items()
    )

    # Spawn robot in Gazebo
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'ball_e',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.1',
        ],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description
        }]
    )

    return LaunchDescription([
        declare_use_sim_time,
        set_gazebo_model_path,
        gazebo,
        robot_state_publisher,
        spawn_robot,
    ])
