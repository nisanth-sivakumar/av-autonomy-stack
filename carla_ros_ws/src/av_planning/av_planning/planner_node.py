#!/usr/bin/env python3
"""
Road Graph Planner Node — av_control package  (Phase 2a)
=========================================================
Subscribes : /carla/ego_vehicle/odometry   (nav_msgs/Odometry)
             /planning/goal                (geometry_msgs/PointStamped)  [optional]
Publishes  : /planning/trajectory          (nav_msgs/Path)
             /planning/debug/trajectory    (nav_msgs/Path)               [latched]

Phase 2 architecture
--------------------
The planner owns the CARLA Python API.  The controller no longer calls it.

                ┌──────────────┐    /planning/trajectory    ┌────────────────┐
  CARLA map ──▶ │ planner_node │ ─────────────────────────▶ │ controller_node│
                └──────────────┘                            └────────────────┘

The planner runs at a fixed rate (default 10 Hz):
  1. Prune waypoints that the vehicle has passed.
  2. If fewer than `replan_threshold` WPs remain → replan from current ego pose.
  3. Append new WPs to the live trajectory (no gap/stutter for the controller).
  4. Publish nav_msgs/Path.

Extension points
----------------
Phase 2b — Lattice planner:
    Override _plan() to generate multiple lateral candidate paths,
    score each, and return the best.

Phase 2b — MPPI:
    Override _plan() to sample N control sequences, roll out trajectories,
    and return the minimum-cost rollout as a Path.

Goal-directed navigation:
    Publish a geometry_msgs/PointStamped on /planning/goal.
    The planner will bias junction choices toward that goal until reached.

Coordinate conventions (identical to controller)
-------------------------------------------------
    ros_x  =  carla_x
    ros_y  = -carla_y
    ros_yaw = -carla_yaw
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, PointStamped

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def make_pose_stamped(x: float, y: float, yaw: float, header) -> PoseStamped:
    ps = PoseStamped()
    ps.header = header
    ps.pose.position.x = x
    ps.pose.position.y = y
    ps.pose.position.z = 0.0
    ps.pose.orientation.z = math.sin(yaw / 2.0)
    ps.pose.orientation.w = math.cos(yaw / 2.0)
    return ps


# ---------------------------------------------------------------------------
# Planner node
# ---------------------------------------------------------------------------

class RoadGraphPlanner(Node):
    """
    Phase 2a baseline planner: walks the CARLA road graph and publishes a
    rolling nav_msgs/Path that the Pure Pursuit controller can track.
    """

    def __init__(self):
        super().__init__("road_graph_planner")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("carla_host",          "localhost")
        self.declare_parameter("carla_port",          2000)
        self.declare_parameter("carla_timeout",       10.0)

        # Road graph walk
        self.declare_parameter("wp_step_m",           3.0)   # spacing between WPs
        self.declare_parameter("wp_lookahead_count",  80)    # WPs per planning call

        # Rolling buffer management
        self.declare_parameter("replan_threshold",    20)    # replan if fewer WPs remain
        self.declare_parameter("max_trajectory_len",  200)   # cap total buffered WPs

        # Goal reaching
        self.declare_parameter("goal_reached_dist",   5.0)   # metres

        # Output
        self.declare_parameter("publish_rate_hz",     10.0)
        self.declare_parameter("frame_id",            "map")

        # ── State ─────────────────────────────────────────────────────────
        self._ego_x:   float | None = None
        self._ego_y:   float | None = None
        self._ego_yaw: float | None = None
        self._ego_speed: float = 0.0

        # The live trajectory buffer: list of (ros_x, ros_y, ros_yaw)
        self._trajectory: list[tuple[float, float, float]] = []

        self._carla_map = None
        self._goal:  tuple[float, float] | None = None
        self._goal_reached = False

        # ── CARLA ─────────────────────────────────────────────────────────
        self._connect_to_carla()

        # ── QoS ───────────────────────────────────────────────────────────
        # Latched: controller always gets the latest path even if it starts late
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        best_effort_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Explicitly match the carla_ros_bridge publisher QoS
        odom_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ── Publishers ────────────────────────────────────────────────────
        self._traj_pub  = self.create_publisher(Path, "/planning/trajectory",       latched_qos)
        self._debug_pub = self.create_publisher(Path, "/planning/debug/trajectory", latched_qos)

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Odometry,      "/carla/ego_vehicle/odometry", self._odom_cb, odom_qos)
        self.create_subscription(PointStamped,  "/planning/goal",              self._goal_cb, 10)

        # ── Planning timer ────────────────────────────────────────────────
        rate = self.get_parameter("publish_rate_hz").value
        self.create_timer(1.0 / rate, self._plan_and_publish)

        self.get_logger().info("RoadGraphPlanner ready — waiting for first odometry…")

    # ── CARLA connection ──────────────────────────────────────────────────

    def _connect_to_carla(self):
        if not CARLA_AVAILABLE:
            self.get_logger().error(
                "CARLA Python API not found on PYTHONPATH.  "
                "Planner will use straight-line fallback."
            )
            return

        host    = self.get_parameter("carla_host").value
        port    = self.get_parameter("carla_port").value
        timeout = self.get_parameter("carla_timeout").value

        try:
            self.get_logger().info(f"Connecting to CARLA at {host}:{port} …")
            client = carla.Client(host, port)
            client.set_timeout(timeout)
            world  = client.get_world()
            self._carla_map = world.get_map()
            self.get_logger().info(f"CARLA connected — map = {self._carla_map.name}")
        except Exception as exc:
            self.get_logger().error(
                f"CARLA connection failed: {exc}\n"
                "  Is CARLA running on {host}:{port}?\n"
                "  Falling back to straight-line trajectory."
            )

    # ── Subscribers ───────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self._ego_x   = msg.pose.pose.position.x
        self._ego_y   = msg.pose.pose.position.y
        self._ego_yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self._ego_speed = math.hypot(
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
        )

    def _goal_cb(self, msg: PointStamped):
        self._goal         = (msg.point.x, msg.point.y)
        self._goal_reached = False
        # Force immediate replan toward new goal
        self._trajectory   = []
        self.get_logger().info(
            f"New goal: ({self._goal[0]:.2f}, {self._goal[1]:.2f})"
        )

    # ── Planning loop ─────────────────────────────────────────────────────

    def _plan_and_publish(self):
        # Wait for first valid odometry
        if self._ego_x is None:
            return
        if abs(self._ego_x) < 0.1 and abs(self._ego_y) < 0.1:
            return   # CARLA startup 0,0 artefact

        # Check goal reached
        if self._goal and not self._goal_reached:
            dist = math.hypot(
                self._ego_x - self._goal[0],
                self._ego_y - self._goal[1],
            )
            if dist < self.get_parameter("goal_reached_dist").value:
                self.get_logger().info(
                    f"Goal reached  dist={dist:.2f} m — holding trajectory."
                )
                self._goal_reached = True

        # Drop waypoints the vehicle has passed
        self._prune_passed_waypoints()

        # Replan when buffer is low
        replan_thresh = self.get_parameter("replan_threshold").value
        max_len       = self.get_parameter("max_trajectory_len").value

        if len(self._trajectory) < replan_thresh:
            self.get_logger().info(f"Replanning — {len(self._trajectory)} WPs remaining.")
            new_wps = self._plan(self._ego_x, self._ego_y, self._ego_yaw)
            # Append and cap length so the buffer never explodes
            self._trajectory.extend(new_wps)
            if len(self._trajectory) > max_len:
                self._trajectory = self._trajectory[:max_len]

        self._publish_trajectory()

    # ── Core planner ─────────────────────────────────────────────────────
    # Replace / extend this method for Phase 2b (Lattice / MPPI)

    def _plan(
        self,
        ros_x: float,
        ros_y: float,
        yaw:   float,
    ) -> list[tuple[float, float, float]]:
        """
        Road-graph baseline planner.

        Snaps to the nearest driveable road waypoint, corrects lane direction,
        then walks the graph forward `wp_lookahead_count` steps at `wp_step_m`
        spacing.  Returns a list of (ros_x, ros_y, ros_yaw) tuples.

        To upgrade to a lattice planner:
            Use this output as the centre-line reference, sample ±N lateral
            offsets per waypoint, and score each candidate trajectory.

        To upgrade to MPPI:
            Replace the road-graph walk with a forward dynamics simulation
            driven by sampled control sequences and a cost function.
        """
        if self._carla_map is None:
            return self._fallback_straight_line(ros_x, ros_y, yaw)

        step  = self.get_parameter("wp_step_m").value
        count = self.get_parameter("wp_lookahead_count").value

        # ── 1. Snap to nearest driving lane ──────────────────────────────
        carla_loc = carla.Location(x=ros_x, y=-ros_y, z=0.5)
        try:
            start_wp = self._carla_map.get_waypoint(
                carla_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception as exc:
            self.get_logger().error(f"get_waypoint failed: {exc}")
            return self._fallback_straight_line(ros_x, ros_y, yaw)

        # ── 2. Correct lane direction (don't follow oncoming traffic) ────
        start_wp = self._correct_lane_direction(start_wp, yaw)

        # ── 3. Walk the road graph ────────────────────────────────────────
        wps = self._walk_graph(start_wp, step, count)

        # ── 4. Validate: first WP must be ahead of the ego ───────────────
        if wps and not self._is_ahead(ros_x, ros_y, yaw, wps[0]):
            self.get_logger().warn(
                "First planned WP is behind — trying opposite lane."
            )
            other = start_wp.get_left_lane() or start_wp.get_right_lane()
            if other and other.lane_type == carla.LaneType.Driving:
                wps = self._walk_graph(other, step, count)

        if wps:
            self.get_logger().info(
                f"Planned {len(wps)} WPs  "
                f"start=({wps[0][0]:.1f}, {wps[0][1]:.1f})  "
                f"end=({wps[-1][0]:.1f}, {wps[-1][1]:.1f})"
            )
        return wps

    # ── Graph-walking helpers ─────────────────────────────────────────────

    def _walk_graph(
        self,
        start_wp,
        step:  float,
        count: int,
    ) -> list[tuple[float, float, float]]:
        """Walk `count` steps of `step` metres along the road graph."""
        waypoints = []
        current   = start_wp
        for _ in range(count):
            nexts = current.next(step)
            if not nexts:
                self.get_logger().warn("Road graph ended — no more waypoints.")
                break
            current = self._choose_next(nexts)
            cx = current.transform.location.x
            cy = -current.transform.location.y   # CARLA → ROS
            # Convert CARLA waypoint yaw to ROS yaw
            cyaw = -math.radians(current.transform.rotation.yaw)
            waypoints.append((cx, cy, cyaw))
        return waypoints

    def _choose_next(self, nexts):
        """
        Junction policy: pick the branch most aligned with the goal direction,
        or default to the first option (roughly straight-ahead).

        Phase 2b: extend with a global route planner (e.g. A* on CARLA's
        topology graph) to pre-compute a route and follow it here.
        """
        if len(nexts) == 1 or self._goal is None:
            return nexts[0]

        # Simple goal-directed heuristic: minimise remaining distance to goal
        gx, gy = self._goal
        return min(
            nexts,
            key=lambda wp: math.hypot(
                wp.transform.location.x - gx,
                -wp.transform.location.y - gy,
            ),
        )

    def _correct_lane_direction(self, wp, ros_yaw: float):
        """
        If the snapped waypoint is on the opposing lane, flip to the correct one.
        """
        wp_yaw_carla  = math.radians(wp.transform.rotation.yaw)
        car_yaw_carla = -ros_yaw
        diff = abs(
            math.atan2(
                math.sin(wp_yaw_carla - car_yaw_carla),
                math.cos(wp_yaw_carla - car_yaw_carla),
            )
        )
        if diff > math.pi / 2:
            other = wp.get_left_lane()
            if other and other.lane_type == carla.LaneType.Driving:
                self.get_logger().debug("Lane direction corrected → left lane.")
                return other
        return wp

    # ── Trajectory maintenance ────────────────────────────────────────────

    def _prune_passed_waypoints(self):
        """Remove waypoints that are now behind the vehicle."""
        hx = math.cos(self._ego_yaw)
        hy = math.sin(self._ego_yaw)
        while self._trajectory:
            wx, wy, _ = self._trajectory[0]
            dx = wx - self._ego_x
            dy = wy - self._ego_y
            if dx * hx + dy * hy > 0.0:
                break
            self._trajectory.pop(0)

    # ── Publisher ─────────────────────────────────────────────────────────

    def _publish_trajectory(self):
        header = self.get_clock().now().to_msg()

        path = Path()
        path.header.stamp    = header
        path.header.frame_id = self.get_parameter("frame_id").value

        for (x, y, yaw) in self._trajectory:
            ps = make_pose_stamped(x, y, yaw, path.header)
            path.poses.append(ps)

        self._traj_pub.publish(path)
        self._debug_pub.publish(path)   # identical copy on debug topic for rviz

    # ── Fallback ──────────────────────────────────────────────────────────

    def _fallback_straight_line(
        self, ros_x: float, ros_y: float, yaw: float
    ) -> list[tuple[float, float, float]]:
        self.get_logger().warn(
            "No CARLA map available — generating straight-line fallback trajectory."
        )
        return [
            (
                ros_x + i * 5.0 * math.cos(yaw),
                ros_y + i * 5.0 * math.sin(yaw),
                yaw,
            )
            for i in range(1, 21)
        ]

    # ── Static helpers ────────────────────────────────────────────────────

    @staticmethod
    def _is_ahead(
        ros_x: float,
        ros_y: float,
        yaw:   float,
        wp:    tuple[float, float, float],
    ) -> bool:
        dx = wp[0] - ros_x
        dy = wp[1] - ros_y
        return dx * math.cos(yaw) + dy * math.sin(yaw) > 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = RoadGraphPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()