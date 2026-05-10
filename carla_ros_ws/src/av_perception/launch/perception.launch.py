# carla_ros_ws/src/av_perception/launch/perception.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='av_perception',
            executable='perception_node',
            name='perception_node',
            parameters=[{
                # CARLA connection
                'carla_host':             '172.25.80.1',
                'carla_port':             2000,
                'carla_timeout':          10.0,

                # Object detection
                'detection_range_m':      50.0,   # detect actors within 50 m
                'min_detection_dist_m':   2.0,    # ignore within 2 m (ego car parts)
                'position_noise_std_m':   0.4,    # Gaussian noise std — simulates sensor uncertainty
                'detection_rate_hz':      10.0,

                # Lane detection
                'lane_roi_top_frac':      0.55,   # ROI starts at 55% down the image
                'pixels_per_meter':       80.0,   # tune: larger → less sensitive offset
                'min_lane_pixels':        20,     # min pixels to trust detection

                # Debug
                'publish_debug_image':    True,   # view on /perception/debug/image in RViz
            }],
            output='screen',
        )
    ])