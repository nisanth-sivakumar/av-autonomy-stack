import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl
import math
import tf_transformations


class SimpleController(Node):
    def __init__(self):
        super().__init__('simple_controller')

        # Publisher
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/ego_vehicle/vehicle_control_cmd',
            10
        )

        # Subscriber
        self.odom_sub = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.odom_callback,
            10
        )

        # State
        self.current_speed = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Control params
        self.target_speed = 5.0  # m/s
        self.kp = 0.5

        # Pure Pursuit params
        self.lookahead_distance = 5.0

        # Simple test path (forces turning)
        self.waypoints = [
            (10, 0),
            (20, 0),
            (30, 5),
            (40, 10),
            (50, 15)
        ]

    def odom_callback(self, msg):
        # Extract position
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Extract velocity
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.sqrt(vx**2 + vy**2)

        # Extract yaw
        orientation = msg.pose.pose.orientation
        _, _, self.current_yaw = tf_transformations.euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])

        self.control_loop()

    def get_target_waypoint(self, x, y):
        for wx, wy in self.waypoints:
            dist = math.sqrt((wx - x)**2 + (wy - y)**2)
            if dist > self.lookahead_distance:
                return wx, wy
        return self.waypoints[-1]

    def compute_steering(self, x, y, yaw, target_x, target_y):
        # Transform to vehicle frame
        dx = target_x - x
        dy = target_y - y

        local_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        local_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy

        if local_x == 0:
            return 0.0

        # Pure pursuit curvature
        curvature = (2.0 * local_y) / (self.lookahead_distance ** 2)

        steer = curvature

        # Clamp steering
        return max(-1.0, min(1.0, steer))

    def control_loop(self):
        # Speed control
        error = self.target_speed - self.current_speed
        throttle = self.kp * error
        throttle = max(0.0, min(1.0, throttle))

        # Steering control
        target_x, target_y = self.get_target_waypoint(
            self.current_x, self.current_y
        )

        steer = self.compute_steering(
            self.current_x,
            self.current_y,
            self.current_yaw,
            target_x,
            target_y
        )

        # Publish control
        control_msg = CarlaEgoVehicleControl()
        control_msg.throttle = throttle
        control_msg.steer = steer
        control_msg.brake = 0.0

        self.control_pub.publish(control_msg)

        # Debug prints
        self.get_logger().info(
            f"Speed: {self.current_speed:.2f} | "
            f"Throttle: {throttle:.2f} | "
            f"Steer: {steer:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = SimpleController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()