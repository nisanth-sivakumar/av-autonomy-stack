#!/usr/bin/env python3
"""
Pure Pursuit Controller Node — av_control package  (Phase 2 — planner-aware)
==============================================================================
Subscribes : /carla/ego_vehicle/odometry       (nav_msgs/Odometry)
             /planning/trajectory              (nav_msgs/Path)
Publishes  : /carla/ego_vehicle/vehicle_control_cmd  (carla_msgs/CarlaEgoVehicleControl)

Phase 2 changes
---------------
The controller no longer connects to the CARLA Python API or generates
waypoints.  That responsibility now belongs entirely to planner_node.py.

                ┌──────────────┐  /planning/trajectory  ┌────────────────┐
  CARLA map ──▶ │ planner_node │ ─────────────────────▶ │ controller_node│
                └──────────────┘                        └────────────────┘

The controller simply converts the incoming nav_msgs/Path into its internal
waypoint list and runs Pure Pursuit as before.

Coordinate conventions
----------------------
    ros_x  =  carla_x
    ros_y  = -carla_y
    ros_yaw = -carla_yaw

CARLA sign convention for steering
-----------------------------------
    positive steer = RIGHT turn
    When local_y > 0 (target is to the LEFT), we want LEFT = negative steer
    → note the negation in the steer calculation below.
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)
from nav_msgs.msg import Odometry, Path
from carla_msgs.msg import CarlaEgoVehicleControl


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


# ---------------------------------------------------------------------------
# Controller node
# ---------------------------------------------------------------------------

class PurePursuitController(Node):

    def __init__(self):
        super().__init__("pure_pursuit_controller")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("lookahead_distance",  6.0)
        self.declare_parameter("target_speed",        3.0)
        self.declare_parameter("kp_speed",            0.5)
        self.declare_parameter("steering_gain",       1.0)
        self.declare_parameter("max_steer",           0.6)
        self.declare_parameter("wheelbase",           2.875)   # Tesla Model 3
        self.declare_parameter("debug_interval",      10)

        # Stuck detection
        self.declare_parameter("stuck_speed_thresh",  0.2)   # m/s
        self.declare_parameter("stuck_timeout",       3.0)   # seconds

        # No-trajectory guard: brake if the planner has been silent this long
        self.declare_parameter("trajectory_timeout",  2.0)   # seconds

        # ── State ─────────────────────────────────────────────────────────
        self._waypoints: list[tuple[float, float]] = []
        self._tick = 0

        self._slow_since:  float | None = None
        self._is_stuck:    bool         = False

        self._last_trajectory_stamp: float | None = None

        # ── QoS: match planner's latched publisher ─────────────────────────
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
            CarlaEgoVehicleControl,
            "/carla/ego_vehicle/vehicle_control_cmd",
            10,
        )
        self.create_subscription(
            Odometry,
            "/carla/ego_vehicle/odometry",
            self._odom_cb,
            odom_qos,
        )
        self.create_subscription(
            Path,
            "/planning/trajectory",
            self._trajectory_cb,
            latched_qos,
        )

        self.get_logger().info(
            "PurePursuitController (Phase 2) ready — "
            "waiting for /planning/trajectory and /carla/ego_vehicle/odometry …"
        )

    # ── Trajectory callback ───────────────────────────────────────────────

    def _trajectory_cb(self, msg: Path):
        """
        Convert nav_msgs/Path → internal waypoint list.

        Only (x, y) is used by the Pure Pursuit law; yaw from the planner
        is stored but not yet consumed (reserved for Phase 3 preview heading).
        """
        self._waypoints = [
            (ps.pose.position.x, ps.pose.position.y)
            for ps in msg.poses
        ]
        self._last_trajectory_stamp = time.monotonic()

        self.get_logger().debug(
            f"Trajectory updated — {len(self._waypoints)} WPs received."
        )

    # ── Odometry callback (main control loop) ─────────────────────────────

    def _odom_cb(self, msg: Odometry):
        px    = msg.pose.pose.position.x
        py    = msg.pose.pose.position.y
        yaw   = quaternion_to_yaw(msg.pose.pose.orientation)
        speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

        # Skip CARLA startup artefact (briefly at 0,0)
        if abs(px) < 0.1 and abs(py) < 0.1:
            return

        self._tick += 1
        debug = (self._tick % self._p("debug_interval") == 0)

        # ── Guard: no trajectory yet ──────────────────────────────────────
        now = time.monotonic()
        if self._last_trajectory_stamp is None:
            if debug:
                self.get_logger().warn(
                    "No trajectory received yet — waiting for planner_node."
                )
            self._publish(0.0, 0.0, 0.0)
            return

        traj_age = now - self._last_trajectory_stamp
        if traj_age > self._p("trajectory_timeout"):
            self.get_logger().warn(
                f"Trajectory stale ({traj_age:.1f} s) — braking. "
                "Is planner_node running?"
            )
            self._publish(0.0, 0.0, 1.0)
            return

        # ── Stuck detection ───────────────────────────────────────────────
        if self._check_stuck(speed, now):
            self._publish(0.0, 0.0, 1.0)
            return

        if debug:
            self.get_logger().info(
                f"[ego]  pos=({px:.2f}, {py:.2f})  "
                f"yaw={math.degrees(yaw):.1f}°  speed={speed:.2f} m/s  "
                f"wps_remaining={len(self._waypoints)}"
            )

        # ── Prune passed waypoints ────────────────────────────────────────
        self._prune_passed(px, py, yaw)

        # ── Find lookahead target ─────────────────────────────────────────
        target = self._find_lookahead(px, py, yaw, debug)
        if target is None:
            if debug:
                self.get_logger().warn(
                    "No lookahead target — trajectory exhausted. "
                    "Waiting for planner to extend it."
                )
            # Don't brake hard; planner will extend shortly
            self._publish(0.0, 0.0, 0.0)
            return

        tx, ty = target

        # ── Body-frame transform ──────────────────────────────────────────
        dx, dy  = tx - px, ty - py
        c,  s   = math.cos(-yaw), math.sin(-yaw)
        local_x =  c * dx - s * dy
        local_y =  s * dx + c * dy

        if debug:
            self.get_logger().info(
                f"[pursuit]  target=({tx:.2f}, {ty:.2f})  "
                f"local=({local_x:.2f}, {local_y:.2f})"
            )

        # ── Pure Pursuit steering ─────────────────────────────────────────
        ld = math.hypot(local_x, local_y)
        if ld < 1e-3:
            steer_raw = 0.0
        else:
            curvature       = (2.0 * local_y) / (ld ** 2)
            L               = self._p("wheelbase")
            steer_angle_rad = math.atan(curvature * L)
            steer_raw       = steer_angle_rad / math.radians(70.0)

        gain  = self._p("steering_gain")
        max_s = self._p("max_steer")
        steer = float(max(-max_s, min(max_s, -steer_raw * gain)))  # note negation

        if debug:
            self.get_logger().info(
                f"[steer]  local_y={local_y:.3f}  raw={steer_raw:.4f}  "
                f"final={steer:.4f}  "
                f"({'RIGHT' if steer > 0 else 'LEFT' if steer < 0 else 'STRAIGHT'})"
            )

        # ── Speed P-controller ────────────────────────────────────────────
        target_speed = self._p("target_speed")
        throttle = float(
            max(0.0, min(1.0, self._p("kp_speed") * (target_speed - speed)))
        )

        if debug:
            self.get_logger().info(
                f"[speed]  target={target_speed:.1f}  "
                f"current={speed:.2f}  throttle={throttle:.3f}"
            )

        self._publish(throttle, steer, 0.0)

    # ── Waypoint helpers ──────────────────────────────────────────────────

    def _prune_passed(self, px: float, py: float, yaw: float):
        """Drop waypoints that are now behind the vehicle."""
        hx = math.cos(yaw)
        hy = math.sin(yaw)
        while self._waypoints:
            wx, wy = self._waypoints[0]
            dx, dy = wx - px, wy - py
            if dx * hx + dy * hy > 0.0:
                break
            self._waypoints.pop(0)

    def _find_lookahead(
        self,
        px: float,
        py: float,
        yaw: float,
        debug: bool = False,
    ) -> tuple[float, float] | None:
        """
        Return the first waypoint at least `lookahead_distance` ahead of ego.
        Falls back to the final waypoint in the buffer if none is far enough.
        """
        ld = self._p("lookahead_distance")

        for (wx, wy) in self._waypoints:
            if math.hypot(wx - px, wy - py) >= ld:
                return (wx, wy)

        # All WPs closer than lookahead (near end of trajectory)
        if self._waypoints:
            return self._waypoints[-1]

        return None

    # ── Stuck detection ───────────────────────────────────────────────────

    def _check_stuck(self, speed: float, now: float) -> bool:
        thresh   = self._p("stuck_speed_thresh")
        timeout  = self._p("stuck_timeout")

        if speed < thresh:
            if self._slow_since is None:
                self._slow_since = now
            elif now - self._slow_since > timeout:
                if not self._is_stuck:
                    self.get_logger().warn(
                        f"STUCK: speed={speed:.2f} m/s for >{timeout:.0f} s. "
                        "Braking. Reset vehicle and relaunch."
                    )
                    self._is_stuck = True
                return True
        else:
            self._slow_since = None
            self._is_stuck   = False

        return False

    # ── Helpers ───────────────────────────────────────────────────────────

    def _p(self, name):
        return self.get_parameter(name).value

    def _publish(self, throttle: float, steer: float, brake: float):
        cmd = CarlaEgoVehicleControl()
        cmd.throttle          = float(throttle)
        cmd.steer             = float(steer)
        cmd.brake             = float(brake)
        cmd.hand_brake        = False
        cmd.reverse           = False
        cmd.manual_gear_shift = False
        self._cmd_pub.publish(cmd)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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