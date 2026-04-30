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
                'carla_host': '172.25.80.1',  
                'target_speed': 3.0,
                'lookahead_distance': 6.0,
                'debug_interval': 10,
            }],
            output='screen',
        )
    ])