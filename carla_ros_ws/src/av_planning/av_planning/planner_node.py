#!/usr/bin/env python3
"""
planner_node.py — Phase 3 update  (Perception-aware planning)
==============================================================
Subscribes : /carla/ego_vehicle/odometry   (nav_msgs/Odometry)
             /planning/goal                (geometry_msgs/PointStamped)
             /perception/obstacles         (geometry_msgs/PoseArray)   ← NEW Phase 3
             /perception/lane_offset       (std_msgs/Float32)          ← NEW Phase 3
Publishes  : /planning/trajectory          (nav_msgs/Path)
             /planning/debug/trajectory    (nav_msgs/Path)

Phase 3 changes from Phase 2b
------------------------------
- Subscribes to /perception/obstacles published by perception_node.
  Obstacle positions are passed directly into the lattice and MPPI
  planners, which already have the obstacle avoidance cost implemented.

- Subscribes to /perception/lane_offset.  The lane offset is used to
  log a warning when the ego drifts significantly from lane centre, and
  is available as a hook for future bias of the MPPI nominal offset.

- No changes to the controller.  The planning/control interface
  (/planning/trajectory) is unchanged.

Coordinate conventions
----------------------
    ros_x = carla_x,  ros_y = -carla_y,  ros_yaw = -carla_yaw
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
from geometry_msgs.msg import PoseStamped, PointStamped, PoseArray
from std_msgs.msg import Float32

from av_planning.lattice_planner import LatticePlanner, LatticeConfig
from av_planning.mppi_planner    import MPPIPlanner, MPPIConfig

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


def make_pose_stamped(x, y, yaw, header) -> PoseStamped:
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

    PLANNER_TYPES = ("road_graph", "lattice", "mppi")

    def __init__(self):
        super().__init__("road_graph_planner")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("carla_host",           "localhost")
        self.declare_parameter("carla_port",           2000)
        self.declare_parameter("carla_timeout",        10.0)

        # Road graph
        self.declare_parameter("wp_step_m",            3.0)
        self.declare_parameter("wp_lookahead_count",   80)

        # Rolling buffer (road_graph + lattice only)
        self.declare_parameter("replan_threshold",     20)
        self.declare_parameter("max_trajectory_len",   200)
        self.declare_parameter("goal_reached_dist",    5.0)

        # Planner selection
        self.declare_parameter("planner_type",         "road_graph")

        # MPPI (road-constrained)
        self.declare_parameter("mppi_num_samples",     400)
        self.declare_parameter("mppi_horizon",         40)
        self.declare_parameter("mppi_temperature",     0.3)
        self.declare_parameter("mppi_lateral_std",     0.2)
        self.declare_parameter("mppi_max_offset",      1.5)
        self.declare_parameter("mppi_update_step",     0.5)
        self.declare_parameter("mppi_w_lane",          3.0)
        self.declare_parameter("mppi_w_smoothness",    5.0)
        self.declare_parameter("mppi_w_deviation",     0.2)
        self.declare_parameter("mppi_w_obstacle",      10.0)

        # Lattice
        self.declare_parameter("lattice_num_offsets",  7)
        self.declare_parameter("lattice_max_offset",   1.8)
        self.declare_parameter("lattice_w_deviation",  1.0)
        self.declare_parameter("lattice_w_curvature",  0.5)
        self.declare_parameter("lattice_w_smoothness", 0.3)

        # Phase 3 — perception integration
        self.declare_parameter("lane_offset_warn_m",   0.5)   # log warn above this drift

        # Output
        self.declare_parameter("publish_rate_hz",      10.0)
        self.declare_parameter("frame_id",             "map")

        # ── Validate planner type ─────────────────────────────────────────
        planner_type = self.get_parameter("planner_type").value
        if planner_type not in self.PLANNER_TYPES:
            self.get_logger().error(
                f"Unknown planner_type='{planner_type}'. "
                f"Valid: {self.PLANNER_TYPES}. Falling back to 'road_graph'."
            )
            planner_type = "road_graph"
        self._planner_type = planner_type

        # ── Sub-planners ──────────────────────────────────────────────────
        self._lattice = LatticePlanner(LatticeConfig(
            num_offsets  = self.get_parameter("lattice_num_offsets").value,
            max_offset_m = self.get_parameter("lattice_max_offset").value,
            w_deviation  = self.get_parameter("lattice_w_deviation").value,
            w_curvature  = self.get_parameter("lattice_w_curvature").value,
            w_smoothness = self.get_parameter("lattice_w_smoothness").value,
        ))

        self._mppi = MPPIPlanner(MPPIConfig(
            num_samples        = self.get_parameter("mppi_num_samples").value,
            horizon            = self.get_parameter("mppi_horizon").value,
            temperature        = self.get_parameter("mppi_temperature").value,
            lateral_std        = self.get_parameter("mppi_lateral_std").value,
            max_lateral_offset = self.get_parameter("mppi_max_offset").value,
            update_step_size   = self.get_parameter("mppi_update_step").value,
            w_lane             = self.get_parameter("mppi_w_lane").value,
            w_smoothness       = self.get_parameter("mppi_w_smoothness").value,
            w_deviation        = self.get_parameter("mppi_w_deviation").value,
            w_obstacle         = self.get_parameter("mppi_w_obstacle").value,
        ))

        # ── State ─────────────────────────────────────────────────────────
        self._ego_x:     float | None = None
        self._ego_y:     float | None = None
        self._ego_yaw:   float | None = None
        self._ego_speed: float        = 0.0
        self._trajectory: list[tuple[float, float, float]] = []
        self._carla_map  = None
        self._goal:      tuple[float, float] | None = None
        self._goal_reached = False

        # Phase 3 perception state
        self._obstacle_positions: list[tuple[float, float]] = []
        self._lane_offset: float = 0.0

        # ── CARLA ─────────────────────────────────────────────────────────
        self._connect_to_carla()

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

        # ── Publishers ────────────────────────────────────────────────────
        self._traj_pub  = self.create_publisher(Path, "/planning/trajectory",       latched_qos)
        self._debug_pub = self.create_publisher(Path, "/planning/debug/trajectory", latched_qos)

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            Odometry, "/carla/ego_vehicle/odometry", self._odom_cb, odom_qos
        )
        self.create_subscription(
            PointStamped, "/planning/goal", self._goal_cb, 10
        )
        # Phase 3: perception inputs
        self.create_subscription(
            PoseArray, "/perception/obstacles", self._obstacles_cb, 10
        )
        self.create_subscription(
            Float32, "/perception/lane_offset", self._lane_offset_cb, 10
        )

        # ── Timer ─────────────────────────────────────────────────────────
        rate = self.get_parameter("publish_rate_hz").value
        self.create_timer(1.0 / rate, self._plan_and_publish)

        self.get_logger().info(
            f"RoadGraphPlanner ready — planner_type='{self._planner_type}'  "
            "waiting for odometry and perception…"
        )

    # ── CARLA connection ──────────────────────────────────────────────────

    def _connect_to_carla(self):
        if not CARLA_AVAILABLE:
            self.get_logger().error("CARLA Python API not found — straight-line fallback.")
            return
        host    = self.get_parameter("carla_host").value
        port    = self.get_parameter("carla_port").value
        timeout = self.get_parameter("carla_timeout").value
        try:
            client = carla.Client(host, port)
            client.set_timeout(timeout)
            world  = client.get_world()
            self._carla_map = world.get_map()
            self.get_logger().info(f"CARLA connected — map={self._carla_map.name}")
        except Exception as exc:
            self.get_logger().error(f"CARLA connection failed: {exc}")

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self._ego_x     = msg.pose.pose.position.x
        self._ego_y     = msg.pose.pose.position.y
        self._ego_yaw   = quaternion_to_yaw(msg.pose.pose.orientation)
        self._ego_speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def _goal_cb(self, msg: PointStamped):
        self._goal         = (msg.point.x, msg.point.y)
        self._goal_reached = False
        self._trajectory   = []
        self._mppi.reset()
        self.get_logger().info(f"New goal: ({self._goal[0]:.2f}, {self._goal[1]:.2f})")

    def _obstacles_cb(self, msg: PoseArray):
        """
        Receive obstacle positions from perception_node.
        Converts PoseArray → list of (x, y) tuples consumed by planners.
        """
        self._obstacle_positions = [
            (p.position.x, p.position.y)
            for p in msg.poses
        ]

    def _lane_offset_cb(self, msg: Float32):
        """
        Receive lane offset from perception_node.
        Positive = ego is LEFT of centre.
        """
        self._lane_offset = msg.data
        warn_threshold = self.get_parameter("lane_offset_warn_m").value
        if abs(self._lane_offset) > warn_threshold:
            self.get_logger().warn(
                f"[lane] Ego drifted {self._lane_offset:+.2f} m from lane centre."
            )

    # ── Planning loop ─────────────────────────────────────────────────────

    def _plan_and_publish(self):
        if self._ego_x is None:
            return
        if abs(self._ego_x) < 0.1 and abs(self._ego_y) < 0.1:
            return

        if self._goal and not self._goal_reached:
            dist = math.hypot(self._ego_x - self._goal[0], self._ego_y - self._goal[1])
            if dist < self.get_parameter("goal_reached_dist").value:
                self.get_logger().info(f"Goal reached (dist={dist:.2f} m).")
                self._goal_reached = True

        if self._planner_type == "mppi":
            # MPPI: receding horizon — always replace, never buffer
            centre_line      = self._get_centre_line(self._ego_x, self._ego_y, self._ego_yaw)
            self._trajectory = self._dispatch_planner(centre_line)
            self.get_logger().info(
                f"[mppi] {len(self._trajectory)} WPs  "
                f"obstacles={len(self._obstacle_positions)}  "
                f"lane_offset={self._lane_offset:+.2f} m"
            )
        else:
            # road_graph / lattice: rolling buffer
            self._prune_passed()
            replan_thresh = self.get_parameter("replan_threshold").value
            max_len       = self.get_parameter("max_trajectory_len").value

            if len(self._trajectory) < replan_thresh:
                self.get_logger().info(
                    f"[{self._planner_type}] Replanning — "
                    f"{len(self._trajectory)} WPs remaining  "
                    f"obstacles={len(self._obstacle_positions)}  "
                    f"lane_offset={self._lane_offset:+.2f} m"
                )
                centre_line = self._get_centre_line(self._ego_x, self._ego_y, self._ego_yaw)
                new_wps     = self._dispatch_planner(centre_line)
                self._trajectory.extend(new_wps)
                if len(self._trajectory) > max_len:
                    self._trajectory = self._trajectory[:max_len]
                self.get_logger().info(
                    f"[{self._planner_type}] Planned {len(new_wps)} WPs  "
                    f"buffer={len(self._trajectory)}"
                )

        self._publish_trajectory()

    # ── Planner dispatcher ────────────────────────────────────────────────

    def _dispatch_planner(
        self,
        centre_line: list[tuple[float, float, float]],
    ) -> list[tuple[float, float, float]]:
        """
        Route to the selected planner, passing live perception data.

        Both lattice and MPPI planners accept obstacle_positions and
        already have the obstacle avoidance cost implemented — this is
        the Phase 3 connection that closes the perception → planning loop.
        """
        if self._planner_type == "lattice":
            return self._lattice.plan(
                centre_line        = centre_line,
                obstacle_positions = self._obstacle_positions,  # ← Phase 3
            )
        elif self._planner_type == "mppi":
            return self._mppi.plan(
                centre_line        = centre_line,
                obstacle_positions = self._obstacle_positions,  # ← Phase 3
            )
        else:
            return centre_line

    # ── Road graph centre-line ────────────────────────────────────────────

    def _get_centre_line(self, ros_x, ros_y, yaw):
        if self._carla_map is None:
            return self._fallback_straight_line(ros_x, ros_y, yaw)

        step  = self.get_parameter("wp_step_m").value
        count = self.get_parameter("wp_lookahead_count").value

        carla_loc = carla.Location(x=ros_x, y=-ros_y, z=0.5)
        try:
            start_wp = self._carla_map.get_waypoint(
                carla_loc, project_to_road=True, lane_type=carla.LaneType.Driving,
            )
        except Exception as exc:
            self.get_logger().error(f"get_waypoint failed: {exc}")
            return self._fallback_straight_line(ros_x, ros_y, yaw)

        start_wp = self._correct_lane_direction(start_wp, yaw)
        wps      = self._walk_graph(start_wp, step, count)

        if wps and not self._is_ahead(ros_x, ros_y, yaw, wps[0]):
            other = start_wp.get_left_lane() or start_wp.get_right_lane()
            if other and other.lane_type == carla.LaneType.Driving:
                wps = self._walk_graph(other, step, count)

        return wps

    def _walk_graph(self, start_wp, step, count):
        wps, current = [], start_wp
        for _ in range(count):
            nexts = current.next(step)
            if not nexts:
                break
            current = self._choose_next(nexts)
            wps.append((
                current.transform.location.x,
                -current.transform.location.y,
                -math.radians(current.transform.rotation.yaw),
            ))
        return wps

    def _choose_next(self, nexts):
        if len(nexts) == 1 or self._goal is None:
            return nexts[0]
        gx, gy = self._goal
        return min(nexts, key=lambda wp: math.hypot(
            wp.transform.location.x - gx,
            -wp.transform.location.y - gy,
        ))

    def _correct_lane_direction(self, wp, ros_yaw):
        wp_yaw  = math.radians(wp.transform.rotation.yaw)
        car_yaw = -ros_yaw
        diff    = abs(math.atan2(math.sin(wp_yaw - car_yaw), math.cos(wp_yaw - car_yaw)))
        if diff > math.pi / 2:
            other = wp.get_left_lane()
            if other and other.lane_type == carla.LaneType.Driving:
                return other
        return wp

    # ── Trajectory maintenance ────────────────────────────────────────────

    def _prune_passed(self):
        if self._ego_yaw is None:
            return
        hx, hy = math.cos(self._ego_yaw), math.sin(self._ego_yaw)
        while self._trajectory:
            wx, wy, _ = self._trajectory[0]
            if (wx - self._ego_x) * hx + (wy - self._ego_y) * hy > 0.0:
                break
            self._trajectory.pop(0)

    # ── Publisher ─────────────────────────────────────────────────────────

    def _publish_trajectory(self):
        path = Path()
        path.header.stamp    = self.get_clock().now().to_msg()
        path.header.frame_id = self.get_parameter("frame_id").value
        for (x, y, yaw) in self._trajectory:
            path.poses.append(make_pose_stamped(x, y, yaw, path.header))
        self._traj_pub.publish(path)
        self._debug_pub.publish(path)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _is_ahead(ros_x, ros_y, yaw, wp):
        return (wp[0] - ros_x) * math.cos(yaw) + (wp[1] - ros_y) * math.sin(yaw) > 0.0

    def _fallback_straight_line(self, ros_x, ros_y, yaw):
        self.get_logger().warn("Using straight-line fallback.")
        return [(ros_x + i * 5.0 * math.cos(yaw),
                 ros_y + i * 5.0 * math.sin(yaw), yaw) for i in range(1, 21)]


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