"""
ROS-free control math used by controller_node.

Everything here operates in the ROS map frame:
    ros_x = carla_x,  ros_y = -carla_y,  ros_yaw = -carla_yaw
"""

import math
from typing import Optional, Sequence

Point = tuple[float, float]


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def prune_passed(waypoints: list[Point], px: float, py: float, yaw: float) -> None:
    """Drop (in place) leading waypoints that are behind the vehicle."""
    hx, hy = math.cos(yaw), math.sin(yaw)
    while waypoints:
        wx, wy = waypoints[0]
        if (wx - px) * hx + (wy - py) * hy > 0.0:
            break
        waypoints.pop(0)


def find_lookahead(
    waypoints: Sequence[Point], px: float, py: float, lookahead: float,
) -> Optional[Point]:
    """Return the first waypoint at least `lookahead` away (else last, else None)."""
    for (wx, wy) in waypoints:
        if math.hypot(wx - px, wy - py) >= lookahead:
            return (wx, wy)
    return waypoints[-1] if waypoints else None


def pure_pursuit_steer(
    px: float, py: float, yaw: float, target: Point,
    wheelbase: float, gain: float, max_steer: float,
) -> float:
    """
    Return the CARLA steer command in [-max_steer, max_steer] for a target.

    CARLA convention: positive steer = RIGHT.  A target to the left
    (local_y > 0) therefore yields a negative command.  The wheel angle is
    normalised by 70 deg (the Model 3's approximate max wheel angle).
    """
    dx, dy = target[0] - px, target[1] - py
    c, s = math.cos(-yaw), math.sin(-yaw)
    local_x = c * dx - s * dy
    local_y = s * dx + c * dy
    ld = math.hypot(local_x, local_y)
    if ld < 1e-3:
        return 0.0
    curvature = 2.0 * local_y / (ld ** 2)
    steer_raw = math.atan(curvature * wheelbase) / math.radians(70.0)
    return float(max(-max_steer, min(max_steer, -steer_raw * gain)))


def obstacle_along_path(
    px: float, py: float, yaw: float,
    waypoints: Sequence[Point],
    obstacles: Sequence[Point],
    corridor_half_width: float,
) -> Optional[float]:
    """
    Return the arc length to the nearest obstacle inside the path corridor.

    Arc length is measured in metres from the ego position along the path.
    An obstacle counts if its distance to the path is within
    `corridor_half_width`; obstacles behind the vehicle are ignored.
    Returns None if the corridor is clear.
    """
    if not obstacles:
        return None
    path = [(px, py)] + list(waypoints)
    if len(path) < 2:
        # No path: fall back to a straight corridor along the heading.
        path.append((px + 50.0 * math.cos(yaw), py + 50.0 * math.sin(yaw)))

    hx, hy = math.cos(yaw), math.sin(yaw)
    best: Optional[float] = None
    for (ox, oy) in obstacles:
        if (ox - px) * hx + (oy - py) * hy <= 0.0:
            continue  # behind ego
        # Closest point on the path polyline → lateral distance and arc length
        s_acc = 0.0
        min_lat, s_at_min = float("inf"), 0.0
        for (ax, ay), (bx, by) in zip(path[:-1], path[1:]):
            abx, aby = bx - ax, by - ay
            seg = math.hypot(abx, aby)
            if seg < 1e-6:
                continue
            t = ((ox - ax) * abx + (oy - ay) * aby) / (seg * seg)
            t = max(0.0, min(1.0, t))
            lat = math.hypot(ox - (ax + t * abx), oy - (ay + t * aby))
            if lat < min_lat:
                min_lat, s_at_min = lat, s_acc + t * seg
            s_acc += seg
        if min_lat <= corridor_half_width and (best is None or s_at_min < best):
            best = s_at_min
    return best


def obstacle_speed_limit(
    obstacle_dist: Optional[float], target_speed: float,
    stop_distance: float, comfort_decel: float,
) -> float:
    """
    Return the speed that allows stopping `stop_distance` short of an obstacle.

    Assumes constant deceleration `comfort_decel`:  v = sqrt(2·a·(s − s_stop)).
    """
    if obstacle_dist is None:
        return target_speed
    free = obstacle_dist - stop_distance
    if free <= 0.0:
        return 0.0
    return min(target_speed, math.sqrt(2.0 * comfort_decel * free))


def speed_command(
    speed: float, speed_limit: float, kp_throttle: float, kp_brake: float,
) -> tuple[float, float]:
    """
    Return (throttle, brake) from a P-law on speed error.

    Accelerates with throttle, coasts for small overspeed, brakes for
    larger overspeed, and holds full brake when commanded to stop.
    """
    if speed_limit <= 0.05:
        return 0.0, 1.0 if speed < 0.5 else min(1.0, max(0.3, kp_brake * speed))
    err = speed_limit - speed
    if err >= 0.0:
        return float(min(1.0, kp_throttle * err)), 0.0
    if err > -0.5:
        return 0.0, 0.0  # coast
    return 0.0, float(min(1.0, kp_brake * (-err)))


class ObstacleSpeedGovernor:
    """
    Obstacle-aware speed limit with noise filtering and restart hysteresis.

    Perceived obstacle positions carry per-frame noise, so the raw
    arc-length distance jitters.  This applies a median over the last
    `window` samples, and once the vehicle has been commanded to stop it
    only releases when the filtered distance exceeds
    `stop_distance + resume_margin` — avoiding stop/creep oscillation at
    the stop line.
    """

    def __init__(self, stop_distance: float, comfort_decel: float,
                 resume_margin: float = 2.0, window: int = 5):
        self.stop_distance = stop_distance
        self.comfort_decel = comfort_decel
        self.resume_margin = resume_margin
        self.window = max(1, int(window))
        self._hist: list[float] = []
        self.holding = False

    def update(self, obstacle_dist: Optional[float], target_speed: float) -> float:
        """Return the speed limit for this tick given the raw obstacle distance."""
        self._hist.append(math.inf if obstacle_dist is None else obstacle_dist)
        self._hist = self._hist[-self.window:]
        filtered = sorted(self._hist)[len(self._hist) // 2]
        dist = None if math.isinf(filtered) else filtered

        if self.holding:
            # Release only when *every* recent sample is clear or beyond the
            # resume line — a single "clear" frame (obstacle flickering at the
            # corridor edge) must not restart the vehicle.
            if min(self._hist) <= self.stop_distance + self.resume_margin:
                return 0.0
            self.holding = False
        limit = obstacle_speed_limit(dist, target_speed, self.stop_distance, self.comfort_decel)
        if limit <= 0.0:
            self.holding = True
        return limit

    @property
    def filtered_distance(self) -> Optional[float]:
        if not self._hist:
            return None
        f = sorted(self._hist)[len(self._hist) // 2]
        return None if math.isinf(f) else f
