# carla_ros_ws/src/av_control/launch/controller.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Follow /clock (CARLA bridge / offline sim)'),
        Node(
            package='av_control',
            executable='controller_node',
            name='pure_pursuit_controller',
            parameters=[{
                'use_sim_time':       LaunchConfiguration('use_sim_time'),
                'target_speed':       3.0,
                'steering_gain':      1.0,
                'debug_interval':     10,

                # Adaptive lookahead: lookahead_time_s of travel, clamped,
                # then capped at curvature_gain x the tightest radius ahead
                'lookahead_time_s':         2.0,    # 6 m at 3 m/s
                'lookahead_min_m':          3.0,
                'lookahead_max_m':          12.0,
                'lookahead_curvature_gain': 0.25,   # CARLA junction turns R≈14 m → L≈3.4 m

                # Obstacle-aware speed (uses /perception/objects boxes + velocity).
                # Distances are to the obstacle BOX: corridor from the path to the
                # box edge (footprints touch below ~0.93 m; planners keep
                # lattice 1.75 / mppi 1.95), stop distance bumper -> box.
                'use_obstacles':          True,
                'require_perception':     True,   # hold while perception is absent or stale
                'corridor_half_width_m':  1.35,
                'stop_distance_m':        3.5,
                'comfort_decel':          2.0,
                'resume_margin_m':        2.0,
                'obstacle_filter_window': 5,
                'obstacle_predict_s':     4.0,
                'heading_check_m':        4.5,    # straight corridor ahead of the actual car
                'follow_time_gap_s':      1.0,    # lead car: gap = stop + T*v_lead
                'follow_gain':            0.5,
                'ego_half_length':        2.40,   # CARLA Model 3 bounding box

                # Stop this far before the end of the path (goal / dead end)
                'end_stop_margin_m':      1.0,

                # Stuck recovery
                'stuck_timeout':          3.0,
                'stuck_recovery_time':    2.0,
                'stuck_max_retries':      3,
            }],
            output='screen',
        ),
    ])
