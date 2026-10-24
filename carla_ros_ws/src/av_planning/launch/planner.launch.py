# carla_ros_ws/src/av_planning/launch/planner.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    planner_type_arg = DeclareLaunchArgument(
        'planner_type',
        default_value='road_graph',
        description='Planner to use: road_graph | lattice | mppi',
    )

    planner_node = Node(
        package='av_planning',
        executable='planner_node',
        name='road_graph_planner',
        parameters=[{
            # CARLA connection
            'carla_host':           '172.25.80.1',
            'carla_port':           2000,

            # Planner selection
            'planner_type':         LaunchConfiguration('planner_type'),

            # Road graph
            'wp_step_m':            3.0,
            'wp_lookahead_count':   80,
            'route_max_deviation_m': 4.0,   # keep following the previous route within this

            # Rolling buffer (road_graph only; lattice/mppi replan every tick)
            'replan_threshold':     20,
            'max_trajectory_len':   200,

            # Road-constrained MPPI (Phase 2b)
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
            'mppi_min_clearance':   2.8,     # m to obstacle centre (controller corridor is 2.2)

            # Lattice (Phase 2b)
            'lattice_num_offsets':  7,
            'lattice_max_offset':   1.8,
            'lattice_w_deviation':  1.0,
            'lattice_w_curvature':  0.5,
            'lattice_w_smoothness': 0.3,
            'lattice_transition_m': 15.0,    # blend distance to target offset
            'lattice_w_clearance':  2.0,
            'lattice_w_consistency': 0.5,    # hysteresis between ticks
            'lattice_min_clearance': 2.6,    # m to obstacle centre → infeasible
            'lattice_comfort_clearance': 3.2,

            # Phase 3 perception
            'lane_offset_warn_m':   0.5,

            # Output
            'publish_rate_hz':      10.0,
        }],
        output='screen',
    )

    return LaunchDescription([planner_type_arg, planner_node])