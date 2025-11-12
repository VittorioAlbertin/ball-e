#!/usr/bin/env python3
"""
Launch file for Ball-e with empty world (testing).

This launch file starts:
1. Gazebo Fortress with the empty world
2. Spawns the Ball-e robot
3. Starts robot_state_publisher
4. Starts ros_gz_bridge
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    # Get package directories
    ball_e_description_dir = get_package_share_directory('ball_e_description')
    ball_e_gazebo_dir = get_package_share_directory('ball_e_gazebo')

    # Paths
    urdf_file = os.path.join(ball_e_description_dir, 'urdf', 'ball_e.urdf.xacro')
    world_file = os.path.join(ball_e_gazebo_dir, 'worlds', 'empty.sdf')
    bridge_config = os.path.join(ball_e_gazebo_dir, 'config', 'bridge_config.yaml')

    # Process xacro to get URDF
    robot_description = xacro.process_file(urdf_file).toxml()

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    # Gazebo Fortress
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_file],
        output='screen'
    )

    # Spawn robot
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'ball_e',
            '-topic', '/robot_description',
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

    # Image bridge
    image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/camera'],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/camera', '/camera/image_raw')
        ]
    )

    return LaunchDescription([
        declare_use_sim_time,
        gazebo,
        robot_state_publisher,
        spawn_robot,
        image_bridge,
    ])
