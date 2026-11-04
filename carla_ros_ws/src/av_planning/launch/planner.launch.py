# carla_ros_ws/src/av_planning/launch/planner.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument('planner_type', default_value='road_graph',
                              description='Planner to use: road_graph | lattice | mppi'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Follow /clock (CARLA bridge / offline sim)'),
        DeclareLaunchArgument('goal_x', default_value='nan',
                              description='Goal x (ROS map frame); nan = no goal'),
        DeclareLaunchArgument('goal_y', default_value='nan',
                              description='Goal y (ROS map frame); nan = no goal'),
    ]

    planner_node = Node(
        package='av_planning',
        executable='planner_node',
        name='road_graph_planner',
        parameters=[{
            'use_sim_time':         LaunchConfiguration('use_sim_time'),

            # CARLA connection
            'carla_host':           '172.25.80.1',
            'carla_port':           2000,

            # Planner selection
            'planner_type':         LaunchConfiguration('planner_type'),

            # Goal: global route (A* over the road topology) and stop there
            'goal_x':               ParameterValue(LaunchConfiguration('goal_x'), value_type=float),
            'goal_y':               ParameterValue(LaunchConfiguration('goal_y'), value_type=float),
            'goal_reached_dist':    3.0,

            # Road graph
            'wp_step_m':            3.0,
            'wp_lookahead_count':   80,
            'route_max_deviation_m': 4.0,   # keep following the previous route within this

            # Rolling buffer (road_graph only; lattice/mppi replan every tick)
            'replan_threshold':     20,
            'max_trajectory_len':   200,

            # Obstacle clearances are from the ego centreline to the obstacle
            # BOX EDGE (footprints touch below ~0.93 m, the ego half-width).
            # Keep: ego half-width (1.08 incl. mirrors) < controller corridor 1.35
            #       < lattice 1.75 < mppi 1.95
            # can't be passed with >= this even at max offset → leave to the controller (= its corridor)
            'obstacle_pass_clearance': 1.35,
            'obstacle_long_margin': 3.0,     # clear this far before/after obstacles too
            'ego_half_length':      2.40,   # CARLA Model 3 bounding box (half-width 1.08 incl. mirrors)

            # Road-constrained MPPI
            'mppi_num_samples':     400,
            'mppi_horizon':         40,
            'mppi_temperature':     0.3,
            'mppi_lateral_std':     0.2,
            'mppi_max_offset':      1.5,
            'mppi_update_step':     0.5,
            'mppi_w_lane':          3.0,
            'mppi_w_smoothness':    5.0,
            'mppi_w_deviation':     0.2,
            'mppi_w_obstacle':      10.0,
            'mppi_min_clearance':   1.95,

            # Lattice
            'lattice_num_offsets':  7,
            'lattice_max_offset':   1.8,
            'lattice_w_deviation':  1.0,
            'lattice_w_curvature':  0.5,
            'lattice_w_smoothness': 0.3,
            'lattice_transition_m': 15.0,    # blend distance to target offset
            'lattice_w_clearance':  2.0,
            'lattice_w_consistency': 0.5,    # hysteresis between ticks
            'lattice_min_clearance': 1.75,   # → infeasible below
            'lattice_comfort_clearance': 2.35,

            # Behavior layer (lattice / mppi): overtake slow or parked leads by
            # changing into a same-direction lane (never an oncoming one)
            'enable_lane_changes':    True,
            'behavior_target_speed':  3.0,    # = controller target_speed
            'behavior_slow_ratio':    0.8,
            'behavior_slow_time_s':   2.0,
            'behavior_prepare_time_s': 0.5,
            'behavior_change_time_s': 5.0,    # change length = clamp(5 s x v, 10, 20) m
            'behavior_min_change_length_m': 10.0,
            'behavior_max_change_length_m': 20.0,
            'behavior_cooldown_s':    5.0,
            'behavior_front_buffer_m': 4.0,
            'behavior_rear_buffer_m': 4.0,
            'behavior_rear_time_gap_s': 1.5,

            # Perception
            'lane_offset_warn_m':   0.5,
            'obstacle_timeout':     1.0,

            # Output (simulated-time rate with use_sim_time)
            'publish_rate_hz':      10.0,
        }],
        output='screen',
    )

    return LaunchDescription(args + [planner_node])
