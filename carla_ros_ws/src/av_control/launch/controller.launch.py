# carla_ros_ws/src/av_control/launch/controller.launch.py
#
#   ros2 launch av_control controller.launch.py                    # Pure Pursuit (Python)
#   ros2 launch av_control controller.launch.py controller:=mpc    # LTV MPC (av_control_cpp)
#
# Both controllers share the longitudinal safety parameters below (the MPC
# node ports the same logic to C++); each adds its own lateral parameters.
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SHARED = {
    'target_speed':       3.0,
    'debug_interval':     10,

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
}

PURE_PURSUIT = {
    'steering_gain':      1.0,
    # Adaptive lookahead: lookahead_time_s of travel, clamped,
    # then capped at curvature_gain x the tightest radius ahead
    'lookahead_time_s':         2.0,    # 6 m at 3 m/s
    'lookahead_min_m':          3.0,
    'lookahead_max_m':          12.0,
    'lookahead_curvature_gain': 0.25,   # CARLA junction turns R≈14 m → L≈3.4 m
}

MPC = {
    'wheelbase':            3.005,  # CARLA Model 3 (rear-axle bicycle model)
    'rear_axle_offset_m':   1.5,    # odometry (vehicle centre) → rear axle
    'mpc_horizon':          25,     # x mpc_dt = 2.5 s
    'mpc_dt':               0.1,
    'mpc_max_steer_rate':   0.8,    # rad/s at the wheel
    'mpc_q_lateral':        2.0,
    'mpc_q_heading':        1.0,
    'mpc_r_steer':          0.5,    # wheel angle beyond the curvature feed-forward
    'mpc_r_steer_rate':     10.0,
    'mpc_terminal_weight':  5.0,
}


def generate_launch_description():
    sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Follow /clock (CARLA bridge / offline sim)'),
        DeclareLaunchArgument('controller', default_value='pure_pursuit',
                              description='pure_pursuit (Python) | mpc (C++, av_control_cpp)'),
        Node(
            package='av_control',
            executable='controller_node',
            name='pure_pursuit_controller',
            parameters=[{**SHARED, **PURE_PURSUIT, **sim_time}],
            output='screen',
            condition=LaunchConfigurationEquals('controller', 'pure_pursuit'),
        ),
        Node(
            package='av_control_cpp',
            executable='mpc_controller_node',
            name='mpc_controller',
            parameters=[{**SHARED, **MPC, **sim_time}],
            output='screen',
            condition=LaunchConfigurationEquals('controller', 'mpc'),
        ),
    ])
