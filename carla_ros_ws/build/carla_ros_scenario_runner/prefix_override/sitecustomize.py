import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/nisanth/carla_ros_ws/install/carla_ros_scenario_runner'
