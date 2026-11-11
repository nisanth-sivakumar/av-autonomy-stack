#!/usr/bin/env python3
"""
planner_node.py — Perception-aware planning
============================================
Subscribes : /carla/ego_vehicle/odometry   (nav_msgs/Odometry)
             /planning/goal                (geometry_msgs/PointStamped)
             /perception/objects           (derived_object_msgs/ObjectArray: boxes + velocity)
             /perception/obstacles         (geometry_msgs/PoseArray, fallback: points)
             /perception/lane_offset       (std_msgs/Float32)
Publishes  : /planning/trajectory          (nav_msgs/Path)
             /planning/debug/trajectory    (nav_msgs/Path)
             /planning/behavior            (std_msgs/String: behavior state, lattice/mppi)

Centre line
-----------
Walks the CARLA road graph (`next(wp_step_m)` × `wp_lookahead_count`).
- Route persistence: each walk restarts from the previous route's
  waypoint at the car while the car is within `route_max_deviation_m` of
  it; re-projecting onto the map every tick is ambiguous inside
  junctions, where lanes of several connections overlap.
- Global route: with a goal (launch params goal_x/goal_y or
  /planning/goal), A* over the lane-segment topology (route_planner.py)
  decides every junction branch, and the walk ends at the goal so the
  published path ends there; the controller stops at the end of its path.
  Without a goal it takes the first branch.

Planning modes (parameter `planner_type`)
------------------------------------------
- road_graph : publishes the centre line.  Rolling buffer, replaced (not
               extended) when fewer than `replan_threshold` waypoints remain.
               Ignores obstacles (the controller still slows/stops for them).
- lattice    : replans every tick; candidates blend from the previous
               plan's offset to sampled target offsets.
- mppi       : replans every tick; warm start shifted by how far the
               centre line's start moved.

Obstacles are boxes with velocity; lattice and MPPI keep clearance to
each box at its predicted position (obstacles.py).

Behavior layer (lattice / mppi, `enable_lane_changes`)
------------------------------------------------------
behavior.py decides the reference lane: keep, follow, or change into a
same-direction neighbour lane (lanes.py) to overtake a slow or parked
lead, with gap acceptance and abort.  During a change the trajectory
planners get the blended centre line (1−w)·origin + w·target; when it
completes, the route switches to the target lane.

Time
----
All timing uses the node clock, which follows /clock when the launch
file sets use_sim_time (the default), so planning runs at
`publish_rate_hz` of *simulated* time whatever the simulator's speed.

Coordinate conventions
----------------------
    ros_x = carla_x,  ros_y = -carla_y,  ros_yaw = -carla_yaw
"""

import math

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
from geometry_msgs.msg import PoseStamped, PointStamped, PoseArray
from std_msgs.msg import Float32, String
from derived_object_msgs.msg import ObjectArray
import numpy as np

from av_planning.behavior import (
    ABORT, CHANGING, BehaviorConfig, BehaviorPlanner, Lane, SAME)
from av_planning.lanes import neighbour_lane, neighbour_waypoints

from av_planning.lattice_planner import LatticePlanner, LatticeConfig
from av_planning.mppi_planner import MPPIPlanner, MPPIConfig
from av_planning.obstacles import (
    Obstacle, DEFAULT_HALF_LENGTH, DEFAULT_HALF_WIDTH, as_obstacle_array)
from av_planning.route_planner import RouteGraph, Segment, end_path_at_goal

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


def wp_key(wp):
    """Lane-segment key of a CARLA waypoint (matches route_planner segments)."""
    return (wp.road_id, wp.section_id, wp.lane_id)


def wp_xy(wp):
    """ROS-frame (x, y) of a CARLA waypoint."""
    return (wp.transform.location.x, -wp.transform.location.y)


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
        # Route persistence: keep walking the previous route while the ego
        # is within this distance of it (re-project onto the map otherwise)
        self.declare_parameter("route_max_deviation_m", 4.0)

        # Rolling buffer (road_graph only)
        self.declare_parameter("replan_threshold",     20)
        self.declare_parameter("max_trajectory_len",   200)

        # Goal (NaN = none). Also settable at runtime on /planning/goal.
        self.declare_parameter("goal_x",               float("nan"))
        self.declare_parameter("goal_y",               float("nan"))
        self.declare_parameter("goal_reached_dist",    3.0)

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
        self.declare_parameter("mppi_min_clearance",   1.95)  # ego centreline → obstacle box edge

        # Lattice
        self.declare_parameter("lattice_num_offsets",  7)
        self.declare_parameter("lattice_max_offset",   1.8)
        self.declare_parameter("lattice_transition_m", 15.0)
        self.declare_parameter("lattice_w_deviation",  1.0)
        self.declare_parameter("lattice_w_curvature",  0.5)
        self.declare_parameter("lattice_w_smoothness", 0.3)
        self.declare_parameter("lattice_w_clearance",  2.0)
        self.declare_parameter("lattice_w_consistency", 0.5)
        self.declare_parameter("lattice_min_clearance", 1.75)
        self.declare_parameter("lattice_comfort_clearance", 2.35)

        # Ego geometry (Tesla Model 3)
        self.declare_parameter("ego_half_length",      2.35)
        # Obstacles that can't be passed with this clearance even at the
        # planner's max offset are left to the controller (= its corridor)
        self.declare_parameter("obstacle_pass_clearance", 1.35)
        # Path must also be laterally clear this far before/after an obstacle
        self.declare_parameter("obstacle_long_margin", 3.0)

        # Behavior layer (lattice / mppi): overtaking by lane change
        self.declare_parameter("enable_lane_changes",    True)
        self.declare_parameter("behavior_target_speed",  3.0)   # = controller target_speed
        self.declare_parameter("behavior_slow_ratio",    0.8)   # lead < ratio × target is slow
        self.declare_parameter("behavior_slow_time_s",   2.0)
        self.declare_parameter("behavior_prepare_time_s", 0.5)
        self.declare_parameter("behavior_change_time_s", 5.0)
        self.declare_parameter("behavior_min_change_length_m", 10.0)
        self.declare_parameter("behavior_max_change_length_m", 20.0)
        self.declare_parameter("behavior_cooldown_s",    5.0)
        self.declare_parameter("behavior_front_buffer_m", 4.0)
        self.declare_parameter("behavior_rear_buffer_m", 4.0)
        self.declare_parameter("behavior_rear_time_gap_s", 1.5)

        # Perception integration
        self.declare_parameter("lane_offset_warn_m",   0.5)   # log warn above this drift
        self.declare_parameter("obstacle_timeout",     1.0)   # s; drop stale obstacle lists

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
        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self._lattice = LatticePlanner(LatticeConfig(
            num_offsets         = p("lattice_num_offsets"),
            max_offset_m        = p("lattice_max_offset"),
            transition_length_m = p("lattice_transition_m"),
            wp_step_m           = p("wp_step_m"),
            w_deviation         = p("lattice_w_deviation"),
            w_curvature         = p("lattice_w_curvature"),
            w_smoothness        = p("lattice_w_smoothness"),
            w_clearance         = p("lattice_w_clearance"),
            w_consistency       = p("lattice_w_consistency"),
            min_clearance_m     = p("lattice_min_clearance"),
            comfort_clearance_m = p("lattice_comfort_clearance"),
            ego_half_length     = p("ego_half_length"),
            pass_clearance_m    = p("obstacle_pass_clearance"),
            obstacle_long_margin_m = p("obstacle_long_margin"),
        ))
        self._mppi = MPPIPlanner(MPPIConfig(
            num_samples        = p("mppi_num_samples"),
            horizon            = p("mppi_horizon"),
            temperature        = p("mppi_temperature"),
            lateral_std        = p("mppi_lateral_std"),
            max_lateral_offset = p("mppi_max_offset"),
            update_step_size   = p("mppi_update_step"),
            w_lane             = p("mppi_w_lane"),
            w_smoothness       = p("mppi_w_smoothness"),
            w_deviation        = p("mppi_w_deviation"),
            w_obstacle         = p("mppi_w_obstacle"),
            min_clearance_m    = p("mppi_min_clearance"),
            ego_half_length    = p("ego_half_length"),
            pass_clearance_m   = p("obstacle_pass_clearance"),
            obstacle_long_margin_m = p("obstacle_long_margin"),
        ))

        self._behavior = BehaviorPlanner(BehaviorConfig(
            target_speed        = p("behavior_target_speed"),
            slow_ratio          = p("behavior_slow_ratio"),
            slow_time_s         = p("behavior_slow_time_s"),
            prepare_time_s      = p("behavior_prepare_time_s"),
            change_time_s       = p("behavior_change_time_s"),
            min_change_length_m = p("behavior_min_change_length_m"),
            change_length_m     = p("behavior_max_change_length_m"),
            cooldown_s          = p("behavior_cooldown_s"),
            front_buffer_m      = p("behavior_front_buffer_m"),
            rear_buffer_m       = p("behavior_rear_buffer_m"),
            rear_time_gap_s     = p("behavior_rear_time_gap_s"),
            ego_half_length     = p("ego_half_length"),
        ))
        self._last_behavior = ""

        # ── State ─────────────────────────────────────────────────────────
        self._ego_x:     float | None = None
        self._ego_y:     float | None = None
        self._ego_yaw:   float | None = None
        self._ego_speed: float        = 0.0
        self._trajectory: list[tuple[float, float, float]] = []
        self._carla_map  = None
        self._route: list = []                 # CARLA waypoints of the last walk
        self._route_graph: RouteGraph | None = None
        self._goal:      tuple[float, float] | None = None   # on the lane centre
        self._goal_key = None
        self._global_route: set = set()        # lane-segment keys to follow
        self._global_route_pending = False
        self._goal_reached = False
        self._last_plan_xy: tuple[float, float] | None = None   # previous centre-line start (MPPI)

        self._objects: list[Obstacle] = []
        self._objects_stamp: float | None = None
        self._points: list[Obstacle] = []
        self._points_stamp: float | None = None
        self._tick = 0
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
        self._behavior_pub = self.create_publisher(String, "/planning/behavior", 10)

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Odometry, "/carla/ego_vehicle/odometry", self._odom_cb, odom_qos)
        self.create_subscription(PointStamped, "/planning/goal", self._goal_cb, 10)
        self.create_subscription(ObjectArray, "/perception/objects", self._objects_cb, 10)
        self.create_subscription(PoseArray, "/perception/obstacles", self._points_cb, 10)
        self.create_subscription(Float32, "/perception/lane_offset", self._lane_offset_cb, 10)

        # ── Goal from parameters ──────────────────────────────────────────
        gx, gy = p("goal_x"), p("goal_y")
        if math.isfinite(gx) and math.isfinite(gy):
            self._set_goal(gx, gy)

        # ── Timer (node clock: simulated time with use_sim_time) ──────────
        self.create_timer(1.0 / p("publish_rate_hz"), self._plan_and_publish)

        self.get_logger().info(
            f"RoadGraphPlanner ready — planner_type='{self._planner_type}'  "
            "waiting for odometry and perception…"
        )

    # ── Time ──────────────────────────────────────────────────────────────

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ── CARLA connection and topology ─────────────────────────────────────

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

    def _build_route_graph(self):
        segments = [
            Segment(key=wp_key(a), start=wp_xy(a), end=wp_xy(b))
            for a, b in self._carla_map.get_topology()
        ]
        self._route_graph = RouteGraph(segments)
        self.get_logger().info(f"Road topology: {len(segments)} lane segments.")

    # ── Goal ──────────────────────────────────────────────────────────────

    def _set_goal(self, gx: float, gy: float):
        """Snap the goal to the nearest driving-lane centre; route on next tick."""
        if self._carla_map is not None:
            wp = self._carla_map.get_waypoint(
                carla.Location(x=gx, y=-gy, z=0.5), project_to_road=True,
                lane_type=carla.LaneType.Driving)
            self._goal = wp_xy(wp)
            self._goal_key = wp_key(wp)
        else:
            self._goal = (gx, gy)
            self._goal_key = None
        self._goal_reached = False
        self._global_route = set()
        self._global_route_pending = True
        self._trajectory = []
        self._route = []
        self._mppi.reset()
        self._lattice.reset()
        self._behavior.reset()
        self.get_logger().info(
            f"New goal: ({gx:.1f}, {gy:.1f}) → lane centre ({self._goal[0]:.1f}, {self._goal[1]:.1f})")

    def _plan_global_route(self):
        """A* from the ego's lane segment to the goal's (needs the ego pose)."""
        self._global_route_pending = False
        if self._carla_map is None or self._goal_key is None:
            return
        if self._route_graph is None:
            self._build_route_graph()
        start = self._carla_map.get_waypoint(
            carla.Location(x=self._ego_x, y=-self._ego_y, z=0.5), project_to_road=True,
            lane_type=carla.LaneType.Driving)
        start = self._correct_lane_direction(start, self._ego_yaw)
        route = self._route_graph.plan(wp_key(start), self._goal_key)
        if route is None:
            self.get_logger().error(
                "No route to the goal on the road graph — taking the closest branch at junctions.")
            return
        self._global_route = set(route)
        self.get_logger().info(f"Global route: {len(route)} lane segments to the goal.")

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self._ego_x     = msg.pose.pose.position.x
        self._ego_y     = msg.pose.pose.position.y
        self._ego_yaw   = quaternion_to_yaw(msg.pose.pose.orientation)
        self._ego_speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def _goal_cb(self, msg: PointStamped):
        self._set_goal(msg.point.x, msg.point.y)

    def _objects_cb(self, msg: ObjectArray):
        """Obstacle boxes with velocity from perception_node."""
        obs = []
        for o in msg.objects:
            dims = list(o.shape.dimensions)
            hl = dims[0] / 2.0 if len(dims) > 0 and dims[0] > 0 else DEFAULT_HALF_LENGTH
            hw = dims[1] / 2.0 if len(dims) > 1 and dims[1] > 0 else DEFAULT_HALF_WIDTH
            obs.append(Obstacle(
                x=o.pose.position.x, y=o.pose.position.y,
                yaw=quaternion_to_yaw(o.pose.orientation),
                half_length=hl, half_width=hw,
                vx=o.twist.linear.x, vy=o.twist.linear.y,
            ))
        self._objects = obs
        self._objects_stamp = self._now()

    def _points_cb(self, msg: PoseArray):
        """Fallback when no /perception/objects: points as default-size static cars."""
        self._points = [Obstacle(x=p.position.x, y=p.position.y) for p in msg.poses]
        self._points_stamp = self._now()

    def _lane_offset_cb(self, msg: Float32):
        """Lane offset from perception_node (+ = ego LEFT of centre). Monitored only."""
        self._lane_offset = msg.data
        if abs(self._lane_offset) > self.get_parameter("lane_offset_warn_m").value:
            self.get_logger().warn(
                f"[lane] Ego drifted {self._lane_offset:+.2f} m from lane centre.",
                throttle_duration_sec=2.0,
            )

    def _current_obstacles(self) -> list[Obstacle]:
        now, timeout = self._now(), self.get_parameter("obstacle_timeout").value
        if self._objects_stamp is not None and now - self._objects_stamp <= timeout:
            return self._objects
        if self._points_stamp is not None and now - self._points_stamp <= timeout:
            return self._points
        return []

    # ── Planning loop ─────────────────────────────────────────────────────

    def _plan_and_publish(self):
        if self._ego_x is None:
            return
        if abs(self._ego_x) < 0.1 and abs(self._ego_y) < 0.1:
            return   # CARLA spawn artefact
        if self._global_route_pending:
            self._plan_global_route()

        if self._goal and not self._goal_reached:
            dist = math.hypot(self._ego_x - self._goal[0], self._ego_y - self._goal[1])
            if dist < self.get_parameter("goal_reached_dist").value and self._ego_speed < 0.3:
                self.get_logger().info(f"Goal reached (dist={dist:.2f} m).")
                self._goal_reached = True

        obstacles = self._current_obstacles()
        if self._planner_type in ("mppi", "lattice"):
            # Receding horizon: re-optimise against the latest obstacles every tick
            centre_line = self._get_centre_line(self._ego_x, self._ego_y, self._ego_yaw)
            if self.get_parameter("enable_lane_changes").value:
                centre_line = self._apply_behavior(centre_line, obstacles)
            self._trajectory = self._dispatch_planner(centre_line, obstacles)
            if self._tick % 10 == 0:
                self.get_logger().info(
                    f"[{self._planner_type}] {len(self._trajectory)} WPs  "
                    f"obstacles={len(obstacles)}  lane_offset={self._lane_offset:+.2f} m"
                )
        else:
            # road_graph: rolling buffer, replaced (not extended) when low.
            self._prune_passed()
            if len(self._trajectory) < self.get_parameter("replan_threshold").value:
                centre_line = self._get_centre_line(self._ego_x, self._ego_y, self._ego_yaw)
                self._trajectory = centre_line[:self.get_parameter("max_trajectory_len").value]
                if self._tick % 10 == 0:
                    self.get_logger().info(
                        f"[road_graph] Replanned — {len(self._trajectory)} WPs  "
                        f"obstacles={len(obstacles)}  lane_offset={self._lane_offset:+.2f} m"
                    )

        self._tick += 1
        self._publish_trajectory()

    # ── Behavior layer ────────────────────────────────────────────────────

    @staticmethod
    def _is_driving(wp) -> bool:
        return wp.lane_type == carla.LaneType.Driving

    def _apply_behavior(self, centre_line, obstacles):
        """Run the behavior layer; return the centre line to plan around."""
        if self._carla_map is None or not centre_line or len(self._route) < 2:
            return centre_line
        route = self._route                     # [start waypoint] + walked waypoints
        lanes = {"ego": Lane(np.asarray(centre_line, float), SAME)}
        for side in ("left", "right"):
            lanes[side], _ = neighbour_lane(route[1:], side, self._is_driving)
        out = self._behavior.update(self._now(), (self._ego_x, self._ego_y), self._ego_speed,
                                    lanes, as_obstacle_array(obstacles))
        self._publish_behavior(out)

        if out.switch_lane:
            new_route = neighbour_waypoints(route, out.switch_lane, self._is_driving)
            if len(new_route) >= 2:
                self._route = new_route
                self._mppi.reset()
                self._lattice.reset()
                self._last_plan_xy = None
                self.get_logger().info(f"[behavior] lane change {out.switch_lane} complete")
                return [(wp_xy(wp)[0], wp_xy(wp)[1], -math.radians(wp.transform.rotation.yaw))
                        for wp in new_route[1:]]
        target = lanes.get(out.target) if out.target else None
        if target is None:
            return centre_line
        ego = np.asarray(centre_line, float)
        n = min(len(ego), len(target.centre))
        pts = ego[:n, :2]
        seg = np.hypot(*np.diff(np.vstack([[self._ego_x, self._ego_y], pts]), axis=0).T)
        w = out.weights(np.cumsum(seg))[:, None]
        xy = (1.0 - w) * pts + w * target.centre[:n, :2]
        dx, dy = np.gradient(xy[:, 0]), np.gradient(xy[:, 1])
        yaw = np.where(np.hypot(dx, dy) > 1e-6, np.arctan2(dy, dx), ego[:n, 2])
        return [(float(x), float(y), float(a)) for (x, y), a in zip(xy, yaw)]

    def _publish_behavior(self, out):
        text = (f"state={out.state} target={out.target or '-'} "
                f"progress={out.progress:.2f} reason={out.reason}")
        self._behavior_pub.publish(String(data=text))
        if out.state != self._last_behavior:
            self.get_logger().info(f"[behavior] {text}")
            self._last_behavior = out.state

    # ── Planner dispatcher ────────────────────────────────────────────────

    def _dispatch_planner(self, centre_line, obstacles):
        """Route to the selected planner with live obstacles and ego state."""
        if not centre_line:
            return []
        ego_xy = (self._ego_x, self._ego_y)
        if self._planner_type == "lattice":
            return self._lattice.plan(
                centre_line        = centre_line,
                obstacle_positions = obstacles,
                current_offset_m   = self._ego_lateral_offset(centre_line),
                ego_xy             = ego_xy,
                ego_speed          = self._ego_speed,
            )
        if self._planner_type == "mppi":
            # Shift the warm start by how far the centre-line's *start* moved
            # (with route persistence it advances in whole-waypoint steps)
            advance_wps = 0.0
            if self._last_plan_xy is not None:
                moved = math.hypot(centre_line[0][0] - self._last_plan_xy[0],
                                   centre_line[0][1] - self._last_plan_xy[1])
                advance_wps = moved / self.get_parameter("wp_step_m").value
            self._last_plan_xy = (centre_line[0][0], centre_line[0][1])
            return self._mppi.plan(
                centre_line        = centre_line,
                obstacle_positions = obstacles,
                advance_wps        = advance_wps,
                ego_xy             = ego_xy,
                ego_speed          = self._ego_speed,
            )
        return centre_line

    def _ego_lateral_offset(self, centre_line) -> float:
        """Signed lateral offset of the ego from the centre-line (+ = left)."""
        x0, y0, yaw0 = centre_line[0]
        return (self._ego_x - x0) * -math.sin(yaw0) + (self._ego_y - y0) * math.cos(yaw0)

    # ── Road graph centre-line ────────────────────────────────────────────

    def _get_centre_line(self, ros_x, ros_y, yaw):
        if self._carla_map is None:
            return self._fallback_straight_line(ros_x, ros_y, yaw)

        step  = self.get_parameter("wp_step_m").value
        count = self.get_parameter("wp_lookahead_count").value

        # Route persistence (see module docstring)
        start = self._route_start(ros_x, ros_y, yaw)
        if start is not None:
            wps = self._walk_graph(start, step, count)
            if wps:
                return wps

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
        """
        Walk `count` steps along the road graph from start_wp.  With a goal,
        stop at the waypoint closest to it and end the path exactly at the
        goal (end_path_at_goal), so the car stops there.
        """
        wps, current = [], start_wp
        route = [start_wp]
        goal = self._goal
        for _ in range(count):
            nexts = current.next(step)
            if not nexts:
                break
            nxt = self._choose_next(nexts)
            if goal is not None:
                d_cur = math.dist(wp_xy(current), goal)
                if d_cur <= step and math.dist(wp_xy(nxt), goal) > d_cur:
                    # current is the closest waypoint to the goal: end the
                    # path at the goal itself, not up to half a step from it
                    wps = end_path_at_goal(wps, goal)
                    break
            current = nxt
            route.append(current)
            x, y = wp_xy(current)
            wps.append((x, y, -math.radians(current.transform.rotation.yaw)))
        self._route = route
        return wps

    def _route_start(self, ros_x, ros_y, yaw):
        """
        Waypoint of the previous route at (or just behind) the ego, or None
        if there is no route or the ego has left it.
        """
        if not self._route:
            return None
        max_dev = self.get_parameter("route_max_deviation_m").value
        if self._behavior.state in (CHANGING, ABORT):
            # Mid lane change the car is up to a lane width off its route by
            # design; re-projecting then snapped the route into the target
            # lane before the change completed (sweep: abort at progress 1.0).
            max_dev += 3.5
        pts = [wp_xy(wp) for wp in self._route]
        dists = [math.hypot(px - ros_x, py - ros_y) for px, py in pts]
        i = min(range(len(pts)), key=dists.__getitem__)
        if dists[i] > max_dev:
            return None
        # start from the closest route point that is not ahead of the ego
        hx, hy = math.cos(yaw), math.sin(yaw)
        if i > 0 and (pts[i][0] - ros_x) * hx + (pts[i][1] - ros_y) * hy > 0.0:
            i -= 1
        return self._route[i]

    def _choose_next(self, nexts):
        """Junction branch: on the global route if there is one, else first / closest to goal."""
        if len(nexts) == 1:
            return nexts[0]
        if self._global_route:
            on_route = [wp for wp in nexts if wp_key(wp) in self._global_route]
            if on_route:
                return on_route[0]
        if self._goal is None:
            return nexts[0]
        return min(nexts, key=lambda wp: math.dist(wp_xy(wp), self._goal))

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
        self.get_logger().warn("Using straight-line fallback.", throttle_duration_sec=5.0)
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
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
