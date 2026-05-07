# carla_ros_ws/src/av_planning/launch/planner.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='av_planning',
            executable='planner_node',
            name='road_graph_planner',
            parameters=[{
                'carla_host':          '172.25.80.1',
                'carla_port':          2000,
                'wp_step_m':           3.0,
                'wp_lookahead_count':  80,
                'replan_threshold':    20,
                'publish_rate_hz':     10.0,
            }],
            output='screen',
        )
    ])