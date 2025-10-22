from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Launch file for Person State Manager.

    This node maintains the world model of all tracked people, combining
    tracking information from the person tracker with identity information
    from face recognition.

    Parameters:
        cleanup_timeout: Seconds before removing inactive persons (default: 5.0)
        publish_rate: Rate to publish person states in Hz (default: 10.0)
    """

    # Declare launch arguments
    cleanup_timeout_arg = DeclareLaunchArgument(
        'cleanup_timeout',
        default_value='5.0',
        description='Seconds before removing persons not seen'
    )

    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='10.0',
        description='Rate to publish person states (Hz)'
    )

    # Person state manager node
    person_state_manager_node = Node(
        package='perception_pkg',
        executable='person_state_manager',
        name='person_state_manager',
        output='screen',
        parameters=[{
            'cleanup_timeout': LaunchConfiguration('cleanup_timeout'),
            'publish_rate': LaunchConfiguration('publish_rate'),
        }]
    )

    return LaunchDescription([
        cleanup_timeout_arg,
        publish_rate_arg,
        person_state_manager_node,
    ])
