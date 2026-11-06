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


def _as_obstacle(o) -> tuple:
    """(x, y) point or (x, y, yaw, half_length, half_width, vx, vy) box."""
    if len(o) == 2:
        return (o[0], o[1], 0.0, 0.0, 0.0, 0.0, 0.0)
    return tuple(o)


def _closest_on_path(path: Sequence[Point], ox: float, oy: float):
    """Return (distance, arc length, unit direction, lateral, + = left) of the closest point."""
    s_acc = 0.0
    best = (float("inf"), 0.0, (1.0, 0.0), 0.0)
    for (ax, ay), (bx, by) in zip(path[:-1], path[1:]):
        abx, aby = bx - ax, by - ay
        seg = math.hypot(abx, aby)
        if seg < 1e-6:
            continue
        t = max(0.0, min(1.0, ((ox - ax) * abx + (oy - ay) * aby) / (seg * seg)))
        d = math.hypot(ox - (ax + t * abx), oy - (ay + t * aby))
        if d < best[0]:
            ux, uy = abx / seg, aby / seg
            best = (d, s_acc + t * seg, (ux, uy), -(ox - ax) * uy + (oy - ay) * ux)
        s_acc += seg
    return best


def obstacle_along_path(
    px: float, py: float, yaw: float,
    waypoints: Sequence[Point],
    obstacles: Sequence,
    corridor_half_width: float,
    ego_half_length: float = 0.0,
    ego_speed: float = 0.0,
    max_predict_s: float = 4.0,
    min_speed: float = 1.0,
    predict_samples: int = 8,
    moving_threshold: float = 0.5,
    heading_check_m: float = 0.0,
) -> Optional[tuple[float, float]]:
    """
    Return (gap, v_along) for the nearest obstacle blocking the path corridor.

    obstacles : (x, y) points or (x, y, yaw, half_length, half_width, vx, vy)
                boxes in the map frame.
    An obstacle blocks if its box comes within `corridor_half_width` of the
    path anywhere along its predicted motion until the ego gets there
    (t = arc length / max(ego_speed, min_speed), capped at `max_predict_s`;
    the swept positions are sampled) — so a pedestrian walking into, or
    across, the lane counts.  Motion is predicted at constant velocity in
    the path's frame (along-path and lateral speed kept): a car following
    its lane round a curve stays in its lane.  A straight-line prediction
    swept oncoming cars on a curve across the ego's lane, and the car braked.

    gap     : arc length from the ego along the path to the obstacle box,
              minus `ego_half_length` (≈ bumper-to-box distance)
    v_along : obstacle velocity along the path direction there
              (> 0 = moving away from the ego, e.g. a lead car)

    heading_check_m > 0 adds a second, short corridor straight ahead of the
    ego's *actual* pose: from the rear of the car to `heading_check_m` beyond
    its front bumper.  The path corridor follows the plan, but the car lags
    it — mid lane change the plan had already left the lead's lane, so the
    controller sped up while the car was still half behind the lead (1.3 m
    bumper gaps offline).  It is bent by the path's *sustained* curvature
    (sustained_curvature): a straight line on a R≈26 m curve drifts ~1.5 m
    outward in 9 m and braked for cars in the outer lane or parked off the
    road; the S of a lane change reverses sign, so it stays straight there.

    Obstacles slower than `moving_threshold` are treated as stationary (so
    velocity noise on parked cars doesn't sweep them into the corridor).
    Obstacles behind the ego are ignored.  Returns None if clear.
    """
    if not obstacles:
        return None
    path = [(px, py)] + list(waypoints)
    if len(path) < 2:
        # No path: fall back to a straight corridor along the heading.
        path.append((px + 50.0 * math.cos(yaw), py + 50.0 * math.sin(yaw)))

    hx, hy = math.cos(yaw), math.sin(yaw)
    kappa = sustained_curvature(path) if heading_check_m > 0.0 else 0.0
    best: Optional[tuple[float, float]] = None
    for o in obstacles:
        ox, oy, oyaw, hl, hw, rvx, rvy = _as_obstacle(o)
        # Slow = parked for the swept prediction, but report the raw speed
        # (zeroing it made a 0.6 m/s lead read 0 in a third of the frames,
        # so the governor flickered between following and stop-and-hold).
        vx, vy = (0.0, 0.0) if math.hypot(rvx, rvy) < moving_threshold else (rvx, rvy)
        _, s0, (ux, uy), lat0 = _closest_on_path(path, ox, oy)
        if s0 <= 0.0:                     # beside/behind the ego: along its heading
            s0 = (ox - px) * hx + (oy - py) * hy
        v_al, v_lat = vx * ux + vy * uy, -vx * uy + vy * ux     # path frame
        rel = oyaw - math.atan2(uy, ux)
        ext_along = hl * abs(math.cos(rel)) + hw * abs(math.sin(rel))
        ext_lat = hl * abs(math.sin(rel)) + hw * abs(math.cos(rel))
        t = min(max_predict_s, max(s0, 0.0) / max(ego_speed, min_speed))
        steps = predict_samples if (vx or vy) else 0
        for k in range(steps + 1):
            tk = t * k / max(steps, 1)
            s_k, lat_k = s0 + v_al * tk, lat0 + v_lat * tk
            if s_k <= 0.0:
                continue  # behind ego
            if abs(lat_k) - ext_lat > corridor_half_width:
                continue
            gap = max(0.0, s_k - ext_along - ego_half_length)
            v_along = rvx * ux + rvy * uy
            if best is None or gap < best[0]:
                best = (gap, v_along)
        if heading_check_m > 0.0:
            # corridor along the ego's own heading, bent by the road's
            # curvature (current position only)
            a = (ox - px) * hx + (oy - py) * hy
            lat = -(ox - px) * hy + (oy - py) * hx - 0.5 * kappa * a * a
            rel = oyaw - (yaw + kappa * a)
            ea = hl * abs(math.cos(rel)) + hw * abs(math.sin(rel))
            el = hl * abs(math.sin(rel)) + hw * abs(math.cos(rel))
            if (a > 0.0 and a - ea <= ego_half_length + heading_check_m
                    and abs(lat) - el <= corridor_half_width):
                gap = max(0.0, a - ea - ego_half_length)
                if best is None or gap < best[0]:
                    best = (gap, rvx * hx + rvy * hy)
    return best


def sustained_curvature(path: Sequence[Point], length: float = 15.0,
                        step: float = 2.5) -> float:
    """
    Return the curvature common to the path's first `length` m (+ = left).

    The path is resampled every `step` m; if every chord turns the same way
    the result is the net heading change over the stretch divided by its
    length, else 0 (a lane change's S, or a straight).
    """
    s_acc = [0.0]
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        s_acc.append(s_acc[-1] + math.hypot(x1 - x0, y1 - y0))
    total = min(length, s_acc[-1])
    n = int(total // step)
    if n < 3:
        return 0.0
    pts, k = [], 0
    for i in range(n + 1):
        target = i * step
        while k < len(path) - 2 and s_acc[k + 1] < target:
            k += 1
        seg = max(s_acc[k + 1] - s_acc[k], 1e-9)
        f = min(1.0, max(0.0, (target - s_acc[k]) / seg))
        pts.append((path[k][0] + f * (path[k + 1][0] - path[k][0]),
                    path[k][1] + f * (path[k + 1][1] - path[k][1])))
    head = [math.atan2(y1 - y0, x1 - x0) for (x0, y0), (x1, y1) in zip(pts, pts[1:])]
    turns = [(h1 - h0 + math.pi) % (2 * math.pi) - math.pi for h0, h1 in zip(head, head[1:])]
    if all(t > 0 for t in turns) or all(t < 0 for t in turns):
        return sum(turns) / (step * len(turns))
    return 0.0


def effective_gap(gap: Optional[float], v_along: float, comfort_decel: float) -> Optional[float]:
    """
    Return the gap credited with a lead vehicle's own stopping distance.

    An obstacle moving away at v can itself stop within v²/(2a), so the ego only needs
    to be able to stop behind where the lead would stop.  Stationary or
    oncoming obstacles get no credit.
    """
    if gap is None:
        return None
    return gap + max(0.0, v_along) ** 2 / (2.0 * comfort_decel)


def path_length_ahead(px: float, py: float, waypoints: Sequence[Point]) -> float:
    """Length of the path from the ego through all remaining waypoints."""
    pts = [(px, py)] + list(waypoints)
    return sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:]))


def path_curvature_ahead(px: float, py: float, waypoints: Sequence[Point],
                         distance: float, min_spacing: float = 1.0) -> float:
    """
    Return the largest |curvature| (1/m) of the path within `distance` of the ego.

    Uses the Menger curvature of consecutive point triples (points closer than
    `min_spacing` are skipped to avoid amplifying noise).
    """
    pts = [(px, py)]
    travelled = 0.0
    for w in waypoints:
        step = math.dist(pts[-1], w)
        if step < min_spacing:
            continue
        pts.append(w)
        travelled += step
        if travelled > distance:
            break
    kappa = 0.0
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        ab, bc, ca = math.dist(a, b), math.dist(b, c), math.dist(c, a)
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if ab * bc * ca > 1e-9:
            kappa = max(kappa, 2.0 * area2 / (ab * bc * ca))
    return kappa


def adaptive_lookahead(
    speed: float, px: float, py: float, waypoints: Sequence[Point],
    time_s: float, min_m: float, max_m: float, curvature_gain: float,
) -> float:
    """
    Return the Pure Pursuit lookahead distance.

    `time_s` seconds of travel, clamped to [min_m, max_m], then capped at
    `curvature_gain` × the radius of the
    tightest curve within reach.  Pure Pursuit cuts corners by roughly
    L²/(2R), so in tight junction turns a shorter lookahead tracks the
    path much more closely; on straights and gentle curves the cap does
    not bind.
    """
    lookahead = max(min_m, min(max_m, time_s * speed))
    if curvature_gain > 0.0:
        kappa = path_curvature_ahead(px, py, waypoints, lookahead + 5.0)
        if kappa > 1e-3:
            lookahead = max(min_m, min(lookahead, curvature_gain / kappa))
    return lookahead


def obstacle_speed_limit(
    obstacle_dist: Optional[float], target_speed: float,
    stop_distance: float, comfort_decel: float,
) -> float:
    """
    Return the speed that allows stopping `stop_distance` short of a point.

    Used for obstacles (distance = gap to the box) and for the end of the
    path (goal / dead end).

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
    Obstacle-aware speed limit with noise filtering, hysteresis and following.

    Perceived obstacle positions and speeds carry per-frame noise, so both
    the gap and the obstacle's speed along the path are median-filtered
    over the last `window` samples.

    Stationary obstacle (filtered speed < `moving_threshold`): stop
    `stop_distance` short, and once stopped hold until every recent sample
    is clear or beyond `stop_distance + resume_margin` — no stop/creep
    oscillation at the stop line.

    Moving lead: track a time-gap following distance
        d_des = stop_distance + follow_time_gap · v_lead
        v     = v_lead + follow_gain · (gap − d_des)
    capped by the braking limit (crediting the lead's own stopping
    distance), and never latched.  Applying the stop-and-hold law to a
    moving lead put the equilibrium right on the stop line, so noise
    triggered stop-and-go (found by the offline `lead` scenario).
    """

    def __init__(self, stop_distance: float, comfort_decel: float,
                 resume_margin: float = 2.0, window: int = 5,
                 follow_time_gap: float = 1.0, follow_gain: float = 0.5,
                 moving_threshold: float = 0.5):
        self.stop_distance = stop_distance
        self.comfort_decel = comfort_decel
        self.resume_margin = resume_margin
        self.window = max(1, int(window))
        self.follow_time_gap = follow_time_gap
        self.follow_gain = follow_gain
        self.moving_threshold = moving_threshold
        self._hist: list[float] = []
        self._speed_hist: list[float] = []
        self.holding = False
        self.following = False   # hysteresis: stays a lead down to half the threshold

    def update(self, obstacle_dist: Optional[float], target_speed: float,
               lead_speed: float = 0.0) -> float:
        """
        Return the speed limit for this tick.

        obstacle_dist : raw gap to the obstacle (None = clear)
        lead_speed    : its speed along the path (> 0 = moving away)
        """
        self._hist.append(math.inf if obstacle_dist is None else obstacle_dist)
        self._hist = self._hist[-self.window:]
        self._speed_hist.append(0.0 if obstacle_dist is None else lead_speed)
        self._speed_hist = self._speed_hist[-self.window:]
        filtered = sorted(self._hist)[len(self._hist) // 2]
        dist = None if math.isinf(filtered) else filtered
        v_lead = sorted(self._speed_hist)[len(self._speed_hist) // 2]

        lead_threshold = self.moving_threshold * (0.5 if self.following else 1.0)
        self.following = dist is not None and v_lead >= lead_threshold
        if self.following:
            self.holding = False
            d_des = self.stop_distance + self.follow_time_gap * v_lead
            v_follow = v_lead + self.follow_gain * (dist - d_des)
            v_safe = obstacle_speed_limit(dist + v_lead ** 2 / (2.0 * self.comfort_decel),
                                          target_speed, self.stop_distance, self.comfort_decel)
            return max(0.0, min(target_speed, v_follow, v_safe))

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
