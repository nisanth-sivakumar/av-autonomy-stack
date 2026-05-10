# carla_ros_ws/src/av_control/launch/controller.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='av_control',
            executable='controller_node',
            name='pure_pursuit_controller',
            parameters=[{
                'target_speed':       3.0,
                'lookahead_distance': 6.0,    # back to original
                'steering_gain':      1.0,    # back to original
                'debug_interval':     10,
            }],
            output='screen',
        )
    ])