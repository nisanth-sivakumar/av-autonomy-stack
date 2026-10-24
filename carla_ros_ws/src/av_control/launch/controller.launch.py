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

                # Obstacle-aware speed (uses /perception/obstacles)
                'use_obstacles':          True,
                'corridor_half_width_m':  2.2,    # < planners' clearance (lattice 2.6, mppi 2.8)
                'stop_distance_m':        8.0,    # centre-to-centre
                'comfort_decel':          2.0,
                'resume_margin_m':        2.0,
                'obstacle_filter_window': 5,

                # Stuck recovery
                'stuck_timeout':          3.0,
                'stuck_recovery_time':    2.0,
                'stuck_max_retries':      3,
            }],
            output='screen',
        )
    ])
