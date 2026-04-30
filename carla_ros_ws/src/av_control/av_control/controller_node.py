#!/usr/bin/env python3
"""
Pure Pursuit Controller Node — av_control package
====================================================
Subscribes : /carla/ego_vehicle/odometry  (nav_msgs/Odometry)
Publishes  : /carla/ego_vehicle/vehicle_control_cmd  (carla_msgs/CarlaEgoVehicleControl)

Changelog v5 — CARLA map waypoints + stuck detection
------------------------------------------------------
PROBLEM: Relative/hardcoded waypoints go straight through walls and fences.
FIX:     Use CARLA Python API to generate waypoints that follow actual roads.
         The car will never be sent off-road again.

PROBLEM: When car hits an obstacle, throttle=1.0 forever.
FIX:     Stuck detector: if speed < threshold for > stuck_timeout seconds,
         apply brakes and log a warning.

Coordinate frame note
---------------------
CARLA uses a LEFT-handed frame (Y points right).
carla_ros_bridge converts odometry to ROS frame (Y points left) by negating Y:
    ros_x = carla_x
    ros_y = -carla_y

So when we read waypoints from the CARLA Python API we must apply the same:
    wp_ros_x = wp.transform.location.x
    wp_ros_y = -wp.transform.location.y

Setup
-----
Make sure the CARLA Python API egg is on your PYTHONPATH, e.g. in ~/.bashrc:
    export PYTHONPATH=$PYTHONPATH:~/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.10-linux-x86_64.egg

Or find the exact egg name with:
    ls ~/CARLA_0.9.15/PythonAPI/carla/dist/
"""

import math
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl

# ---------------------------------------------------------------------------
# CARLA Python API import — graceful fallback if not on PYTHONPATH
# ---------------------------------------------------------------------------
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class PurePursuitController(Node):

    def __init__(self):
        super().__init__("pure_pursuit_controller")

        # ---- ROS parameters ------------------------------------------------
        self.declare_parameter("carla_host",          "localhost")
        self.declare_parameter("carla_port",          2000)
        self.declare_parameter("carla_timeout",       10.0)
        self.declare_parameter("wp_step_m",           3.0)   # distance between road waypoints
        self.declare_parameter("wp_lookahead_count",  80)    # how many road WPs to generate ahead
        self.declare_parameter("lookahead_distance",  6.0)
        self.declare_parameter("target_speed",        3.0)
        self.declare_parameter("kp_speed",            0.5)
        self.declare_parameter("steering_gain",       1.0)
        self.declare_parameter("max_steer",           0.6)
        self.declare_parameter("debug_interval",      10)
        self.declare_parameter("wheelbase",           2.875)
        self.declare_parameter("stuck_speed_thresh",  0.2)   # m/s — below this = potentially stuck
        self.declare_parameter("stuck_timeout",       3.0)   # seconds below threshold before "stuck"

        # ---- State ---------------------------------------------------------
        self._waypoints     = []
        self._initialised   = False
        self._tick          = 0
        self._laps          = 0
        self._carla_map     = None

        # Stuck detection
        self._slow_since    = None   # timestamp when speed first dropped below threshold
        self._is_stuck      = False

        # ---- CARLA connection ----------------------------------------------
        self._connect_to_carla()

        # ---- ROS pub/sub ---------------------------------------------------
        self.cmd_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            "/carla/ego_vehicle/vehicle_control_cmd",
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            "/carla/ego_vehicle/odometry",
            self._odom_callback,
            10,
        )

        self.get_logger().info("PurePursuitController ready — waiting for first odometry…")

    # ------------------------------------------------------------------
    # CARLA connection
    # ------------------------------------------------------------------
    def _connect_to_carla(self):
        if not CARLA_AVAILABLE:
            self.get_logger().error(
                "CARLA Python API not found on PYTHONPATH.\n"
                "  Add the egg to ~/.bashrc:\n"
                "  export PYTHONPATH=$PYTHONPATH:~/CARLA_0.9.15/PythonAPI/carla/dist/"
                "<carla_egg_name>.egg\n"
                "  Then rebuild: colcon build --packages-select av_control\n"
                "  Falling back to no map waypoints — car will drive straight."
            )
            return

        host    = self.get_parameter("carla_host").value
        port    = self.get_parameter("carla_port").value
        timeout = self.get_parameter("carla_timeout").value

        try:
            self.get_logger().info(f"Connecting to CARLA at {host}:{port}…")
            client = carla.Client(host, port)
            client.set_timeout(timeout)
            world  = client.get_world()
            self._carla_map = world.get_map()
            self.get_logger().info(
                f"Connected to CARLA  map={self._carla_map.name}"
            )
        except Exception as e:
            self.get_logger().error(
                f"Failed to connect to CARLA: {e}\n"
                "  Is CARLA running? Is the port correct?\n"
                "  Falling back to no map waypoints."
            )

    # ------------------------------------------------------------------
    # Waypoint generation from CARLA map
    # ------------------------------------------------------------------
    def _generate_road_waypoints(self, ros_x: float, ros_y: float):
        """
        Walk the CARLA road graph forward from the vehicle's position
        and return a list of (ros_x, ros_y) pairs that follow the road.

        Coordinate conversion: CARLA Y is negated to get ROS Y.
        """
        if self._carla_map is None:
            self.get_logger().warn(
                "No CARLA map available — generating straight-line fallback waypoints."
            )
            return self._straight_line_fallback(ros_x, ros_y)

        step   = self.get_parameter("wp_step_m").value
        count  = self.get_parameter("wp_lookahead_count").value

        # Convert ROS position back to CARLA frame (negate Y)
        carla_loc = carla.Location(x=ros_x, y=-ros_y, z=0.5)

        try:
            start_wp = self._carla_map.get_waypoint(
                carla_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
        except Exception as e:
            self.get_logger().error(f"get_waypoint failed: {e}")
            return self._straight_line_fallback(ros_x, ros_y)

        waypoints = []
        current   = start_wp

        for _ in range(count):
            nexts = current.next(step)
            if not nexts:
                self.get_logger().warn("Road graph ended — no more waypoints ahead.")
                break
            # next() can return multiple options at junctions; take the first
            # (straight-ish) one. Later in Phase 2 a planner will choose.
            current = nexts[0]
            carla_x = current.transform.location.x
            carla_y = current.transform.location.y
            # Apply ROS frame conversion
            waypoints.append((carla_x, -carla_y))

        self.get_logger().info(
            f"Generated {len(waypoints)} road waypoints  "
            f"start=({waypoints[0][0]:.1f},{waypoints[0][1]:.1f})  "
            f"end=({waypoints[-1][0]:.1f},{waypoints[-1][1]:.1f})"
        )
        return waypoints

    def _straight_line_fallback(self, ros_x: float, ros_y: float):
        """Emergency fallback: 20 waypoints straight ahead (no map needed)."""
        self.get_logger().warn(
            "Using straight-line fallback — may drive off-road. "
            "Fix the CARLA Python API connection."
        )
        # This reuses whatever yaw the car had at first odom tick,
        # but we don't have yaw here — so just go along X axis.
        # Caller (_init_waypoints) passes yaw separately; this is best effort.
        return [(ros_x + i * 5.0, ros_y) for i in range(1, 21)]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_waypoints(self, px: float, py: float, yaw: float):
        self._waypoints   = self._generate_road_waypoints(px, py)
        self._initialised = True
        self._slow_since  = None
        self._is_stuck    = False

    # ------------------------------------------------------------------
    # Stuck detection
    # ------------------------------------------------------------------
    def _check_stuck(self, speed: float) -> bool:
        thresh   = self.get_parameter("stuck_speed_thresh").value
        timeout  = self.get_parameter("stuck_timeout").value
        now      = time.monotonic()

        if speed < thresh:
            if self._slow_since is None:
                self._slow_since = now
            elif now - self._slow_since > timeout:
                if not self._is_stuck:
                    self.get_logger().warn(
                        f"STUCK: speed={speed:.2f} m/s for >{timeout:.0f}s. "
                        "Applying brakes. Reset the car and relaunch the controller."
                    )
                    self._is_stuck = True
                return True
        else:
            # Moving again — clear stuck state
            self._slow_since = None
            self._is_stuck   = False

        return False

    # ------------------------------------------------------------------
    # Main control callback
    # ------------------------------------------------------------------
    def _odom_callback(self, msg: Odometry):
        px    = msg.pose.pose.position.x
        py    = msg.pose.pose.position.y
        yaw   = quaternion_to_yaw(msg.pose.pose.orientation)
        speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

        if not self._initialised:
            self._init_waypoints(px, py, yaw)
            return

        self._tick += 1
        debug = (self._tick % self._param("debug_interval") == 0)

        # ---- Stuck check ------------------------------------------------
        if self._check_stuck(speed):
            self._publish(0.0, 0.0, 1.0)
            return

        if debug:
            self.get_logger().info(
                f"[ego] pos=({px:.2f}, {py:.2f})  "
                f"yaw={math.degrees(yaw):.1f}°  speed={speed:.2f} m/s  "
                f"wps_remaining={len(self._waypoints)}"
            )

        # ---- Lookahead waypoint -----------------------------------------
        target = self._find_lookahead_waypoint(px, py, yaw, debug)

        if target is None:
            # End of road segment — regenerate from current position
            self.get_logger().info("End of waypoints — regenerating road waypoints…")
            self._init_waypoints(px, py, yaw)
            target = self._find_lookahead_waypoint(px, py, yaw, debug)

        if target is None:
            self.get_logger().warn("Still no waypoint after regeneration — braking.")
            self._publish(0.0, 0.0, 1.0)
            return

        tx, ty = target

        # ---- Body-frame transform ----------------------------------------
        dx, dy = tx - px, ty - py
        c, s   = math.cos(-yaw), math.sin(-yaw)
        local_x =  c * dx - s * dy
        local_y =  s * dx + c * dy

        if debug:
            self.get_logger().info(
                f"[pursuit] target=({tx:.2f},{ty:.2f})  "
                f"local_x={local_x:.2f}  local_y={local_y:.2f}"
            )

        # ---- Pure Pursuit steering ---------------------------------------
        # CARLA sign convention: positive steer = RIGHT turn
        # When local_y > 0 (target to LEFT) we want LEFT = negative steer
        ld = math.hypot(local_x, local_y)
        if ld < 1e-3:
            steer_raw = 0.0
        else:
            curvature       = (2.0 * local_y) / (ld ** 2)
            L               = self._param("wheelbase")
            steer_angle_rad = math.atan(curvature * L)
            steer_raw       = steer_angle_rad / math.radians(70.0)

        gain  = self._param("steering_gain")
        max_s = self._param("max_steer")
        steer = float(max(-max_s, min(max_s, -steer_raw * gain)))  # note negation

        if debug:
            self.get_logger().info(
                f"[steer] local_y={local_y:.3f}  raw={steer_raw:.4f}  "
                f"final={steer:.4f}  "
                f"({'RIGHT' if steer > 0 else 'LEFT' if steer < 0 else 'STRAIGHT'})"
            )

        # ---- Speed P controller -----------------------------------------
        target_speed = self._param("target_speed")
        throttle     = float(max(0.0, min(1.0, self._param("kp_speed") * (target_speed - speed))))

        if debug:
            self.get_logger().info(
                f"[speed] target={target_speed:.1f}  "
                f"current={speed:.2f}  throttle={throttle:.3f}"
            )

        self._publish(throttle, steer, 0.0)

    # ------------------------------------------------------------------
    # Waypoint selection — consume waypoints as we pass them
    # ------------------------------------------------------------------
    def _find_lookahead_waypoint(self, px, py, yaw, debug=False):
        """
        Drop any waypoints that are now behind the vehicle, then return
        the first one that is at least lookahead_distance ahead.

        Consuming passed waypoints keeps the list from growing stale and
        ensures the car never chases a point it has already passed.
        """
        ld = self._param("lookahead_distance")
        hx = math.cos(yaw)
        hy = math.sin(yaw)

        # Drop waypoints that are now behind us
        while self._waypoints:
            wx, wy = self._waypoints[0]
            dx, dy = wx - px, wy - py
            if dx * hx + dy * hy > 0.0:   # still ahead
                break
            self._waypoints.pop(0)         # behind — discard

        # Find first waypoint beyond lookahead distance
        for (wx, wy) in self._waypoints:
            dx, dy = wx - px, wy - py
            if math.hypot(dx, dy) >= ld:
                return (wx, wy)

        # All remaining waypoints are closer than lookahead (near end of list)
        if self._waypoints:
            return self._waypoints[-1]     # aim at the last one

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _param(self, name):
        return self.get_parameter(name).value

    def _publish(self, throttle, steer, brake):
        cmd = CarlaEgoVehicleControl()
        cmd.throttle = float(throttle)
        cmd.steer    = float(steer)
        cmd.brake    = float(brake)
        cmd.hand_brake        = False
        cmd.reverse           = False
        cmd.manual_gear_shift = False
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()