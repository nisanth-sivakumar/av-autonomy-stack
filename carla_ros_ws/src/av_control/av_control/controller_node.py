#!/usr/bin/env python3
"""
Pure Pursuit controller node for the av_control package.

Subscribes : /carla/ego_vehicle/odometry       (nav_msgs/Odometry)
             /planning/trajectory              (nav_msgs/Path, latched)
             /perception/obstacles             (geometry_msgs/PoseArray)
Publishes  : /carla/ego_vehicle/vehicle_control_cmd  (carla_msgs/CarlaEgoVehicleControl)

                ┌──────────────┐  /planning/trajectory  ┌────────────────┐
  CARLA map ──▶ │ planner_node │ ─────────────────────▶ │ controller_node│
                └──────────────┘                        └────────────────┘
                                   /perception/obstacles ──────▲

Lateral   : Pure Pursuit on the planned path (see pure_pursuit.py).
Longitudinal:
  - target_speed, reduced so the vehicle can stop `stop_distance` short
    of any obstacle inside a corridor around the planned path
    (v = sqrt(2·a·(s − s_stop))).  The planners shift the path laterally
    around obstacles; this is the fallback when the path stays blocked.
    The distance is median-filtered (perception noise) and a stop is held
    until the obstacle is `resume_margin` beyond the stop line.
  - P-law on throttle; braking when over the limit.

Safety guards
-------------
  - zero output until the first trajectory arrives
  - full brake if the trajectory is older than `trajectory_timeout`
  - stuck detection: if throttle is being applied but the vehicle has not
    moved for `stuck_timeout`, brake for `stuck_recovery_time` and retry;
    after `stuck_max_retries` consecutive attempts, hold the brake.
    Deliberate stops (obstacle ahead) never count as stuck.

Coordinate conventions
----------------------
    ros_x = carla_x,  ros_y = -carla_y,  ros_yaw = -carla_yaw
    CARLA steering: positive = RIGHT (see pure_pursuit_steer).
"""

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseArray
from carla_msgs.msg import CarlaEgoVehicleControl

from av_control.pure_pursuit import (
    quaternion_to_yaw,
    prune_passed,
    find_lookahead,
    pure_pursuit_steer,
    obstacle_along_path,
    speed_command,
    ObstacleSpeedGovernor,
)


class PurePursuitController(Node):

    def __init__(self):
        super().__init__("pure_pursuit_controller")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("lookahead_distance",  6.0)
        self.declare_parameter("target_speed",        3.0)
        self.declare_parameter("kp_speed",            0.5)
        self.declare_parameter("kp_brake",            0.5)
        self.declare_parameter("steering_gain",       1.0)
        self.declare_parameter("max_steer",           0.6)
        self.declare_parameter("wheelbase",           2.875)   # Tesla Model 3
        self.declare_parameter("debug_interval",      10)

        # Obstacle-aware speed
        self.declare_parameter("use_obstacles",          True)
        # corridor = ego half-width + obstacle half-width + margin
        # (footprints overlap below ~1.8 m; planners keep a wider berth)
        self.declare_parameter("corridor_half_width_m",  2.2)
        self.declare_parameter("stop_distance_m",        8.0)  # centre-to-centre standoff
        self.declare_parameter("comfort_decel",          2.0)  # m/s²
        self.declare_parameter("resume_margin_m",        2.0)  # extra clearance to restart
        self.declare_parameter("obstacle_filter_window", 5)    # median over N control ticks
        self.declare_parameter("obstacle_timeout",       1.0)  # s; drop stale obstacle lists

        # Stuck detection / recovery
        self.declare_parameter("stuck_speed_thresh",  0.2)   # m/s
        self.declare_parameter("stuck_timeout",       3.0)   # s
        self.declare_parameter("stuck_recovery_time", 2.0)   # s of braking before retry
        self.declare_parameter("stuck_max_retries",   3)

        # No-trajectory guard: brake if the planner has been silent this long
        self.declare_parameter("trajectory_timeout",  2.0)   # s

        # ── State ─────────────────────────────────────────────────────────
        self._waypoints: list[tuple[float, float]] = []
        self._tick = 0
        self._last_trajectory_stamp: float | None = None

        self._obstacles: list[tuple[float, float]] = []
        self._last_obstacle_stamp: float | None = None
        self._governor = ObstacleSpeedGovernor(
            stop_distance=self._p("stop_distance_m"),
            comfort_decel=self._p("comfort_decel"),
            resume_margin=self._p("resume_margin_m"),
            window=self._p("obstacle_filter_window"),
        )

        self._pushing_since: float | None = None   # throttle>0 but not moving
        self._recover_until: float | None = None
        self._stuck_retries = 0

        # ── QoS ───────────────────────────────────────────────────────────
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        odom_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ── ROS pub/sub ───────────────────────────────────────────────────
        self._cmd_pub = self.create_publisher(
            CarlaEgoVehicleControl, "/carla/ego_vehicle/vehicle_control_cmd", 10,
        )
        self.create_subscription(
            Odometry, "/carla/ego_vehicle/odometry", self._odom_cb, odom_qos,
        )
        self.create_subscription(
            Path, "/planning/trajectory", self._trajectory_cb, latched_qos,
        )
        self.create_subscription(
            PoseArray, "/perception/obstacles", self._obstacles_cb, 10,
        )

        self.get_logger().info(
            "PurePursuitController ready — waiting for /planning/trajectory "
            "and /carla/ego_vehicle/odometry …"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _trajectory_cb(self, msg: Path):
        self._waypoints = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        self._last_trajectory_stamp = time.monotonic()

    def _obstacles_cb(self, msg: PoseArray):
        self._obstacles = [(p.position.x, p.position.y) for p in msg.poses]
        self._last_obstacle_stamp = time.monotonic()

    def _odom_cb(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

        # Skip CARLA startup artefact (briefly at 0,0)
        if abs(px) < 0.1 and abs(py) < 0.1:
            return

        self._tick += 1
        debug = (self._tick % self._p("debug_interval") == 0)
        now = time.monotonic()

        # ── Guard: no / stale trajectory ──────────────────────────────────
        if self._last_trajectory_stamp is None:
            if debug:
                self.get_logger().warn("No trajectory received yet — waiting for planner_node.")
            self._publish(0.0, 0.0, 0.0)
            return
        traj_age = now - self._last_trajectory_stamp
        if traj_age > self._p("trajectory_timeout"):
            if debug:
                self.get_logger().warn(
                    f"Trajectory stale ({traj_age:.1f} s) — braking. Is planner_node running?"
                )
            self._publish(0.0, 0.0, 1.0)
            return

        # ── Guard: stuck recovery in progress ─────────────────────────────
        if self._recover_until is not None:
            if now < self._recover_until:
                self._publish(0.0, 0.0, 1.0)
                return
            self._recover_until = None
            self._pushing_since = None
            self.get_logger().info(
                f"Stuck recovery: retrying ({self._stuck_retries}/{self._p('stuck_max_retries')})."
            )

        # ── Lateral: Pure Pursuit ─────────────────────────────────────────
        prune_passed(self._waypoints, px, py, yaw)
        target = find_lookahead(self._waypoints, px, py, self._p("lookahead_distance"))
        if target is None:
            if debug:
                self.get_logger().warn("Trajectory exhausted — waiting for planner to extend it.")
            self._publish(0.0, 0.0, 0.3 if speed > 0.5 else 0.0)
            return
        steer = pure_pursuit_steer(
            px, py, yaw, target,
            wheelbase=self._p("wheelbase"),
            gain=self._p("steering_gain"),
            max_steer=self._p("max_steer"),
        )

        # ── Longitudinal: obstacle-aware speed ────────────────────────────
        target_speed = self._p("target_speed")
        obs_dist = None
        if self._p("use_obstacles") and self._obstacles_fresh(now):
            obs_dist = obstacle_along_path(
                px, py, yaw, self._waypoints, self._obstacles,
                self._p("corridor_half_width_m"),
            )
        speed_limit = self._governor.update(obs_dist, target_speed)
        obs_dist = self._governor.filtered_distance
        throttle, brake = speed_command(
            speed, speed_limit, self._p("kp_speed"), self._p("kp_brake"),
        )

        # ── Stuck detection (only while actually pushing) ─────────────────
        if self._check_stuck(throttle, speed, now):
            self._publish(0.0, 0.0, 1.0)
            return

        if debug:
            obs_txt = f"{obs_dist:.1f} m" if obs_dist is not None else "clear"
            self.get_logger().info(
                f"[ctrl] pos=({px:.1f},{py:.1f}) yaw={math.degrees(yaw):.0f}° "
                f"v={speed:.2f}/{speed_limit:.2f} m/s  steer={steer:+.3f}  "
                f"thr={throttle:.2f} brk={brake:.2f}  obstacle={obs_txt}  "
                f"wps={len(self._waypoints)}"
            )

        self._publish(throttle, steer, brake)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _obstacles_fresh(self, now: float) -> bool:
        return (
            self._last_obstacle_stamp is not None
            and now - self._last_obstacle_stamp <= self._p("obstacle_timeout")
        )

    def _check_stuck(self, throttle: float, speed: float, now: float) -> bool:
        """Return True (and start a recovery phase) if pushing without moving."""
        if speed >= self._p("stuck_speed_thresh"):
            self._pushing_since = None
            self._stuck_retries = 0
            return False
        if throttle <= 0.05:
            self._pushing_since = None   # deliberate stop / coasting
            return False
        if self._pushing_since is None:
            self._pushing_since = now
            return False
        if now - self._pushing_since <= self._p("stuck_timeout"):
            return False

        if self._stuck_retries >= self._p("stuck_max_retries"):
            if self._tick % self._p("debug_interval") == 0:
                self.get_logger().error(
                    "STUCK: retries exhausted — holding brake. Reset the vehicle and relaunch."
                )
            return True
        self._stuck_retries += 1
        self._recover_until = now + self._p("stuck_recovery_time")
        self.get_logger().warn(
            f"STUCK: speed={speed:.2f} m/s with throttle applied for "
            f">{self._p('stuck_timeout'):.0f} s — braking "
            f"{self._p('stuck_recovery_time'):.0f} s then retrying."
        )
        return True

    def _p(self, name):
        return self.get_parameter(name).value

    def _publish(self, throttle: float, steer: float, brake: float):
        cmd = CarlaEgoVehicleControl()
        cmd.throttle = float(throttle)
        cmd.steer = float(steer)
        cmd.brake = float(brake)
        cmd.hand_brake = False
        cmd.reverse = False
        cmd.manual_gear_shift = False
        self._cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
