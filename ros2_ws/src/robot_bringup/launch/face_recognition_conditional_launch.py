from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Launch file for Conditional Face Recognition Node.

    This node performs face recognition on-demand instead of every frame,
    significantly improving performance by only processing when triggered.

    Parameters:
        recognition_threshold: Minimum cosine similarity for face matching (default: 0.6)
        min_face_size: Minimum face size in pixels (default: 20)
        max_face_size: Maximum face size in pixels (default: 400)
        frame_cache_size: Number of frames to cache (default: 10)
        reidentification_interval: Seconds before re-identifying known persons (default: 30.0)
        auto_identify_new_tracks: Automatically identify new tracks (default: true)
    """

    # Declare launch arguments
    recognition_threshold_arg = DeclareLaunchArgument(
        'recognition_threshold',
        default_value='0.6',
        description='Minimum cosine similarity for face matching'
    )

    min_face_size_arg = DeclareLaunchArgument(
        'min_face_size',
        default_value='20',
        description='Minimum face size in pixels'
    )

    max_face_size_arg = DeclareLaunchArgument(
        'max_face_size',
        default_value='400',
        description='Maximum face size in pixels'
    )

    frame_cache_size_arg = DeclareLaunchArgument(
        'frame_cache_size',
        default_value='10',
        description='Number of frames to cache for async processing'
    )

    reidentification_interval_arg = DeclareLaunchArgument(
        'reidentification_interval',
        default_value='30.0',
        description='Seconds before re-identifying known persons'
    )

    auto_identify_new_tracks_arg = DeclareLaunchArgument(
        'auto_identify_new_tracks',
        default_value='true',
        description='Automatically identify new tracks'
    )

    # Face recognition conditional node
    face_recognition_node = Node(
        package='perception_pkg',
        executable='face_recognition_conditional',
        name='face_recognition_conditional',
        output='screen',
        parameters=[{
            'recognition_threshold': LaunchConfiguration('recognition_threshold'),
            'min_face_size': LaunchConfiguration('min_face_size'),
            'max_face_size': LaunchConfiguration('max_face_size'),
            'frame_cache_size': LaunchConfiguration('frame_cache_size'),
            'reidentification_interval': LaunchConfiguration('reidentification_interval'),
            'auto_identify_new_tracks': LaunchConfiguration('auto_identify_new_tracks'),
        }]
    )

    return LaunchDescription([
        recognition_threshold_arg,
        min_face_size_arg,
        max_face_size_arg,
        frame_cache_size_arg,
        reidentification_interval_arg,
        auto_identify_new_tracks_arg,
        face_recognition_node,
    ])
