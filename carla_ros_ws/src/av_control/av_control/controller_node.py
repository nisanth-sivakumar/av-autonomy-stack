#!/usr/bin/env python3
"""
Pure Pursuit controller node for the av_control package.

Subscribes : /carla/ego_vehicle/odometry       (nav_msgs/Odometry)
             /planning/trajectory              (nav_msgs/Path, latched)
             /perception/objects               (derived_object_msgs/ObjectArray: boxes + velocity)
             /perception/obstacles             (geometry_msgs/PoseArray, fallback: points)
Publishes  : /carla/ego_vehicle/vehicle_control_cmd  (carla_msgs/CarlaEgoVehicleControl)

                ┌──────────────┐  /planning/trajectory  ┌────────────────┐
  CARLA map ──▶ │ planner_node │ ─────────────────────▶ │ controller_node│
                └──────────────┘                        └────────────────┘
                                   /perception/objects ────────▲

Lateral
-------
Pure Pursuit on the planned path with an adaptive lookahead: `lookahead_time_s`
of travel clamped to [lookahead_min_m, lookahead_max_m], capped at
`lookahead_curvature_gain` × the radius of the tightest curve ahead (Pure
Pursuit cuts corners by ~L²/2R, which mattered at tight junction turns).

Longitudinal
------------
speed limit = min(target_speed, obstacle limit, end-of-path limit)
  - obstacle limit: stop `stop_distance_m` short of the nearest obstacle box
    whose current or predicted position is within `corridor_half_width_m` of
    the planned path (v = sqrt(2·a·(gap − stop))).  A moving lead vehicle is
    followed at a time gap (gap = stop + T·v_lead) instead, capped by the
    braking limit credited with the lead's own stopping distance.  The
    planners steer around obstacles; this is the fallback.
    The gap is median-filtered and a stop is held until the obstacle is clear
    (ObstacleSpeedGovernor).
  - end-of-path limit: stop `end_stop_margin_m` before the last waypoint.  Normal
    horizons are 60 m+ so this only binds at the goal (the planner ends the path
    there) or a dead end.
  - P-law on throttle; braking when over the limit.

Safety guards
-------------
  - zero output until the first trajectory arrives
  - full brake if the trajectory is older than `trajectory_timeout`
  - stuck detection: if throttle is being applied but the vehicle has not
    moved for `stuck_timeout`, brake for `stuck_recovery_time` and retry;
    after `stuck_max_retries` consecutive attempts, hold the brake.
    Deliberate stops never count as stuck.

Time: the node clock follows /clock with use_sim_time (launch default).

Coordinate conventions
----------------------
    ros_x = carla_x,  ros_y = -carla_y,  ros_yaw = -carla_yaw
    CARLA steering: positive = RIGHT (see pure_pursuit_steer).
"""

import math
import time
from typing import Optional

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
from derived_object_msgs.msg import ObjectArray
from std_msgs.msg import Float32

from av_control.pure_pursuit import (
    quaternion_to_yaw,
    prune_passed,
    find_lookahead,
    pure_pursuit_steer,
    adaptive_lookahead,
    obstacle_along_path,
    path_length_ahead,
    obstacle_speed_limit,
    speed_command,
    ObstacleSpeedGovernor,
)

# Size assumed for obstacles reported as bare points (a mid-size car)
DEFAULT_HALF_LENGTH = 2.2
DEFAULT_HALF_WIDTH = 0.9


class PurePursuitController(Node):

    def __init__(self):
        super().__init__("pure_pursuit_controller")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("target_speed",        3.0)
        self.declare_parameter("kp_speed",            0.5)
        self.declare_parameter("kp_brake",            0.5)
        self.declare_parameter("steering_gain",       1.0)
        self.declare_parameter("max_steer",           0.6)
        self.declare_parameter("wheelbase",           2.875)   # Tesla Model 3
        self.declare_parameter("ego_half_length",     2.35)    # Tesla Model 3: 4.7 m
        self.declare_parameter("debug_interval",      10)

        # Adaptive lookahead
        self.declare_parameter("lookahead_time_s",         2.0)   # 6 m at 3 m/s
        self.declare_parameter("lookahead_min_m",          3.0)
        self.declare_parameter("lookahead_max_m",          12.0)
        self.declare_parameter("lookahead_curvature_gain", 0.25)  # L ≤ 0.25·R

        # Obstacle-aware speed (box edges; footprints touch below ~0.93 m)
        self.declare_parameter("use_obstacles",          True)
        self.declare_parameter("corridor_half_width_m",  1.35)  # path → obstacle box edge
        self.declare_parameter("stop_distance_m",        3.5)   # bumper → box gap when stopped
        self.declare_parameter("comfort_decel",          2.0)   # m/s²
        self.declare_parameter("resume_margin_m",        2.0)   # extra clearance to restart
        self.declare_parameter("obstacle_filter_window", 5)     # median over N control ticks
        self.declare_parameter("obstacle_timeout",       1.0)   # s; drop stale obstacle lists
        # no (fresh) perception = unknown, not clear: hold until it arrives
        self.declare_parameter("require_perception",     True)
        self.declare_parameter("obstacle_predict_s",     4.0)   # max prediction horizon
        # straight corridor ahead of the actual pose (the car lags the plan)
        self.declare_parameter("heading_check_m",        4.5)
        self.declare_parameter("follow_time_gap_s",      1.0)   # lead car: gap = stop + T·v_lead
        self.declare_parameter("follow_gain",            0.5)   # 1/s, gap-error → speed

        # End of path (goal / dead end)
        self.declare_parameter("end_stop_margin_m",   1.0)

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

        self._objects: list[tuple] = []
        self._objects_stamp: float | None = None
        self._points: list[tuple] = []
        self._points_stamp: float | None = None
        self._governor = ObstacleSpeedGovernor(
            stop_distance=self._p("stop_distance_m"),
            comfort_decel=self._p("comfort_decel"),
            resume_margin=self._p("resume_margin_m"),
            window=self._p("obstacle_filter_window"),
            follow_time_gap=self._p("follow_time_gap_s"),
            follow_gain=self._p("follow_gain"),
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
        # per-tick compute time (wall clock: a CPU cost, not node timing),
        # for comparison with av_control_cpp's MPC solve time
        self._time_pub = self.create_publisher(Float32, "/control/compute_time_ms", 10)
        self.create_subscription(Odometry, "/carla/ego_vehicle/odometry", self._odom_cb, odom_qos)
        self.create_subscription(Path, "/planning/trajectory", self._trajectory_cb, latched_qos)
        self.create_subscription(ObjectArray, "/perception/objects", self._objects_cb, 10)
        self.create_subscription(PoseArray, "/perception/obstacles", self._points_cb, 10)

        self.get_logger().info(
            "PurePursuitController ready — waiting for /planning/trajectory "
            "and /carla/ego_vehicle/odometry …"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _trajectory_cb(self, msg: Path):
        self._waypoints = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        self._last_trajectory_stamp = self._now()

    def _objects_cb(self, msg: ObjectArray):
        objs = []
        for o in msg.objects:
            dims = list(o.shape.dimensions)
            hl = dims[0] / 2.0 if len(dims) > 0 and dims[0] > 0 else DEFAULT_HALF_LENGTH
            hw = dims[1] / 2.0 if len(dims) > 1 and dims[1] > 0 else DEFAULT_HALF_WIDTH
            objs.append((o.pose.position.x, o.pose.position.y,
                         quaternion_to_yaw(o.pose.orientation), hl, hw,
                         o.twist.linear.x, o.twist.linear.y))
        self._objects = objs
        self._objects_stamp = self._now()

    def _points_cb(self, msg: PoseArray):
        self._points = [(p.position.x, p.position.y, 0.0, DEFAULT_HALF_LENGTH,
                         DEFAULT_HALF_WIDTH, 0.0, 0.0) for p in msg.poses]
        self._points_stamp = self._now()

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
        now = self._now()

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

        # ── Lateral: Pure Pursuit with adaptive lookahead ─────────────────
        t_compute = time.perf_counter()
        prune_passed(self._waypoints, px, py, yaw)
        lookahead = adaptive_lookahead(
            speed, px, py, self._waypoints,
            time_s=self._p("lookahead_time_s"),
            min_m=self._p("lookahead_min_m"),
            max_m=self._p("lookahead_max_m"),
            curvature_gain=self._p("lookahead_curvature_gain"),
        )
        target = find_lookahead(self._waypoints, px, py, lookahead)
        if target is None:
            # No path ahead (goal reached / road end / planner gap): hold still
            if debug:
                self.get_logger().info("No path ahead — holding.")
            self._publish(0.0, 0.0, 0.3 if speed > 0.5 else 1.0)
            return
        steer = pure_pursuit_steer(
            px, py, yaw, target,
            wheelbase=self._p("wheelbase"),
            gain=self._p("steering_gain"),
            max_steer=self._p("max_steer"),
        )
        self._time_pub.publish(Float32(data=(time.perf_counter() - t_compute) * 1e3))

        # ── Longitudinal ──────────────────────────────────────────────────
        target_speed = self._p("target_speed")
        decel = self._p("comfort_decel")
        hit = None
        perception_ok = True
        if self._p("use_obstacles"):
            obstacles = self._current_obstacles(now)
            perception_ok = obstacles is not None or not self._p("require_perception")
            hit = obstacle_along_path(
                px, py, yaw, self._waypoints, obstacles or [],
                corridor_half_width=self._p("corridor_half_width_m"),
                ego_half_length=self._p("ego_half_length"),
                ego_speed=speed,
                max_predict_s=self._p("obstacle_predict_s"),
                heading_check_m=self._p("heading_check_m"),
            )
        obstacle_limit = self._governor.update(
            hit[0] if hit else None, target_speed, lead_speed=hit[1] if hit else 0.0)
        end_limit = obstacle_speed_limit(
            path_length_ahead(px, py, self._waypoints), target_speed,
            self._p("end_stop_margin_m"), decel,
        )
        speed_limit = min(obstacle_limit, end_limit)
        if not perception_ok:
            # Found by the randomized sweep: perception came up 8 s late, the
            # empty obstacle list read as "clear" and the car hit the lead.
            speed_limit = 0.0
            self.get_logger().warn(
                "No fresh /perception/objects — holding.", throttle_duration_sec=2.0)
        throttle, brake = speed_command(
            speed, speed_limit, self._p("kp_speed"), self._p("kp_brake"),
        )

        # ── Stuck detection (only while actually pushing) ─────────────────
        if self._check_stuck(throttle, speed, now):
            self._publish(0.0, 0.0, 1.0)
            return

        if debug:
            g = self._governor.filtered_distance
            obs_txt = f"{g:.1f} m" if g is not None else "clear"
            self.get_logger().info(
                f"[ctrl] pos=({px:.1f},{py:.1f}) yaw={math.degrees(yaw):.0f}° "
                f"v={speed:.2f}/{speed_limit:.2f} m/s  steer={steer:+.3f}  "
                f"thr={throttle:.2f} brk={brake:.2f}  obstacle={obs_txt}  "
                f"L={lookahead:.1f} m  wps={len(self._waypoints)}"
            )

        self._publish(throttle, steer, brake)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _current_obstacles(self, now: float) -> Optional[list[tuple]]:
        """Return the latest obstacle list, or None if perception is absent or stale."""
        timeout = self._p("obstacle_timeout")
        if self._objects_stamp is not None and now - self._objects_stamp <= timeout:
            return self._objects
        if self._points_stamp is not None and now - self._points_stamp <= timeout:
            return self._points
        return None

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
