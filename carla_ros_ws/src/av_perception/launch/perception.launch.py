# carla_ros_ws/src/av_perception/launch/perception.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Follow /clock (CARLA bridge / offline sim)'),
        Node(
            package='av_perception',
            executable='perception_node',
            name='perception_node',
            parameters=[{
                'use_sim_time':           LaunchConfiguration('use_sim_time'),

                # CARLA connection
                'carla_host':             '172.25.80.1',
                'carla_port':             2000,
                'carla_timeout':          10.0,

                # Object detection
                'detection_range_m':      50.0,   # detect actors within 50 m
                'min_detection_dist_m':   2.0,    # ignore within 2 m (ego car parts)
                'position_noise_std_m':   0.4,    # Gaussian noise std — simulates sensor uncertainty
                'velocity_noise_std_mps': 0.2,
                'detection_rate_hz':      10.0,

                # Lane detection: geometry of semantic_segmentation_front
                # in config/objects.json (z=2.0, pitch=20 down, fov=90).
                'camera_height_m':        2.0,
                'camera_pitch_deg':       20.0,   # positive = looking down
                'camera_fov_deg':         90.0,   # fallback until camera_info arrives
                'lane_max_range_m':       15.0,
                'lane_width_m':           3.5,
                'lane_min_rows':          6,      # plausibility gate (lane_detection.py)
                'lane_max_lateral_rate':  1.5,    # m/s; faster jumps are bad frames
                'lane_max_offset_m':      1.25,   # beyond: wrong lines tracked → no detection
                'lane_hold_time_s':       1.0,

                'publish_debug_image':    True,   # view on /perception/debug/image in RViz
            }],
            output='screen',
        )
    ])