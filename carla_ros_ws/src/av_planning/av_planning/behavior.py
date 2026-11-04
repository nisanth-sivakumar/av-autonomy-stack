"""
behavior.py — behavior layer: lane keeping, following and lane changes (ROS-free).

Sits on top of the lattice / MPPI trajectory planners and decides *which
lane* they should plan in.  Its output is a reference: the current lane,
or a blend from the current lane to a neighbouring one during a lane
change; the planner node turns that into the centre line the trajectory
planners optimise around.

States
------
    LANE_KEEP       no relevant lead in the lane
    FOLLOW          lead in the lane within `follow_range_m` (the controller
                    follows it)
    PREPARE_CHANGE  the lead has been slow (or parked) for `slow_time_s` and a
                    same-direction neighbour lane exists; waiting for the
                    target-lane gap to be acceptable for `prepare_time_s`
    CHANGING        blending to the target lane over L = change_time_s of travel,
                    clamped to [min_change_length_m, change_length_m]
    ABORT           blending back to the origin lane (the target gap closed
                    before the car was `abort_progress_max` of the way across)

A completed change makes the target lane the new current lane (the
planner switches its route to it); the car stays there.  After a change or
an abort, no new change starts for `cooldown_s` (hysteresis).

Rules
-----
- Never change into an oncoming lane (direction −1).
- Gap acceptance: every obstacle in the target lane, predicted at constant
  velocity along the lane over the manoeuvre window, must stay at least
      ahead : lengths + front_buffer_m + front_time_gap_s · v_ego
      behind: lengths + rear_buffer_m  + rear_time_gap_s  · v_obstacle
  away from the ego at every sampled instant.  Re-checked every tick while
  changing; a failure before `abort_progress_max` aborts.

Lane-change profile
-------------------
The blend weight toward the target at arc length s ahead of the ego is
    w(s) = smoothstep((d + s) / L)
where d is the distance travelled since the change started, so the
profile is fixed in space from tick to tick (re-anchoring it at the car
each tick would chase the controller's tracking error — the lattice
"kerb walk" lesson).  An abort blends back from the progress reached.

Frames: ROS map frame; lanes are given as (x, y, yaw) centre lines that
start near the ego and run forward.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

LANE_KEEP = "LANE_KEEP"
FOLLOW = "FOLLOW"
PREPARE_CHANGE = "PREPARE_CHANGE"
CHANGING = "CHANGING"
ABORT = "ABORT"

SAME, ONCOMING = 1, -1


@dataclass
class Lane:
    centre: np.ndarray            # (N, 3) x, y, yaw from near the ego forward
    direction: int = SAME         # SAME as the ego lane, or ONCOMING
    width: float = 3.5


@dataclass
class BehaviorConfig:
    target_speed: float = 3.0
    follow_range_m: float = 30.0
    slow_ratio: float = 0.8           # lead slower than this × target_speed is "slow"
    slow_time_s: float = 2.0
    prepare_time_s: float = 0.5
    change_length_m: float = 20.0     # max lane-change length
    min_change_length_m: float = 10.0
    change_time_s: float = 5.0        # L = clamp(change_time_s · v, min, max)
    abort_length_m: float = 12.0
    abort_progress_max: float = 0.5
    cooldown_s: float = 5.0
    front_buffer_m: float = 4.0
    front_time_gap_s: float = 1.0
    rear_buffer_m: float = 4.0
    rear_time_gap_s: float = 1.5
    min_window_s: float = 4.0         # gap-check horizon at least this long
    ego_half_length: float = 2.4
    moving_threshold: float = 0.5
    # A car counts as in a lane only if its box intrudes this far into it
    # (0.4 m position noise put a lead in the next lane ~1 % of frames).
    lane_overlap_margin_m: float = 0.3
    abort_confirm_ticks: int = 3      # consecutive gap conflicts before aborting
    prefer: tuple = ("left", "right")


@dataclass
class BehaviorOutput:
    state: str
    target: Optional[str] = None      # 'left' / 'right' while CHANGING or ABORT
    switch_lane: Optional[str] = None  # set on the tick a change completes
    progress: float = 0.0             # blend weight at the ego (0 origin … 1 target)
    reason: str = ""
    # profile parameters (see weights())
    _d: float = 0.0
    _length: float = 1.0
    _w0: float = 0.0
    _aborting: bool = False

    def weights(self, s: np.ndarray) -> np.ndarray:
        """Blend weight toward the target lane at arc lengths `s` ahead of the ego."""
        s = np.asarray(s, float)
        if self.target is None:
            return np.zeros_like(s)
        u = np.clip((self._d + s) / max(self._length, 1e-6), 0.0, 1.0)
        ramp = 3 * u ** 2 - 2 * u ** 3
        if self._aborting:
            return self._w0 * (1.0 - ramp)
        return ramp


def smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 3 * u ** 2 - 2 * u ** 3


# ── Geometry helpers ─────────────────────────────────────────────────────────

def project(centre: np.ndarray, x: float, y: float) -> tuple[float, float, float]:
    """
    (arc length, signed lateral offset (+ left), lane heading) of a point
    projected onto a centre-line polyline.  Points before the start are
    measured along the first segment's extension.
    """
    pts = centre[:, :2]
    if len(pts) == 1:
        yaw = centre[0, 2]
        dx, dy = x - pts[0, 0], y - pts[0, 1]
        return (dx * np.cos(yaw) + dy * np.sin(yaw),
                -dx * np.sin(yaw) + dy * np.cos(yaw), yaw)
    a, b = pts[:-1], pts[1:]
    ab = b - a
    seg = np.maximum(np.hypot(ab[:, 0], ab[:, 1]), 1e-9)
    t = ((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / seg ** 2
    t_c = np.clip(t, 0.0, 1.0)
    cx, cy = a[:, 0] + t_c * ab[:, 0], a[:, 1] + t_c * ab[:, 1]
    i = int(np.argmin(np.hypot(x - cx, y - cy)))
    if i == 0 and t[0] < 0.0:
        t_c = t                                     # extend backwards
    ux, uy = ab[i, 0] / seg[i], ab[i, 1] / seg[i]
    s0 = float(np.concatenate([[0.0], np.cumsum(seg)])[i])
    s = s0 + float(t_c[i]) * seg[i]
    px, py = a[i, 0] + t_c[i] * ab[i, 0], a[i, 1] + t_c[i] * ab[i, 1]
    lat = -(x - px) * uy + (y - py) * ux
    return s, float(lat), float(np.arctan2(uy, ux))


def _in_lane(lane: Lane, obs_row: np.ndarray, lat: float, lane_yaw: float,
             margin: float = 0.0) -> bool:
    """Does the obstacle box intrude at least `margin` into the lane laterally?"""
    rel = obs_row[2] - lane_yaw
    ext_lat = obs_row[3] * abs(np.sin(rel)) + obs_row[4] * abs(np.cos(rel))
    return abs(lat) - ext_lat < lane.width / 2.0 - margin


def lead_in_lane(lane: Lane, ego_xy, obs: np.ndarray, cfg: BehaviorConfig):
    """
    Nearest obstacle ahead in the lane within follow_range_m:
    (bumper gap, speed along the lane) or None.
    """
    s_e, _, _ = project(lane.centre, *ego_xy)
    best = None
    for row in obs:
        s_o, lat, yaw = project(lane.centre, row[0], row[1])
        if not _in_lane(lane, row, lat, yaw, cfg.lane_overlap_margin_m):
            continue
        rel = row[2] - yaw
        ext_along = row[3] * abs(np.cos(rel)) + row[4] * abs(np.sin(rel))
        gap = s_o - s_e - ext_along - cfg.ego_half_length
        if s_o <= s_e or gap > cfg.follow_range_m:
            continue
        v_along = row[5] * np.cos(yaw) + row[6] * np.sin(yaw)
        if best is None or gap < best[0]:
            best = (float(gap), float(v_along))
    return best


def target_lane_clear(lane: Lane, ego_xy, ego_speed: float, obs: np.ndarray,
                      window_s: float, cfg: BehaviorConfig) -> tuple[bool, str]:
    """Gap acceptance for the target lane over the manoeuvre window (see module docstring)."""
    if lane.direction != SAME:
        return False, "oncoming lane"
    s_e0, _, _ = project(lane.centre, *ego_xy)
    v_e = max(ego_speed, 1.0)
    ts = np.linspace(0.0, window_s, 25)
    for row in obs:
        s_o0, lat, yaw = project(lane.centre, row[0], row[1])
        if not _in_lane(lane, row, lat, yaw, cfg.lane_overlap_margin_m):
            continue
        v_o = row[5] * np.cos(yaw) + row[6] * np.sin(yaw)
        if np.hypot(row[5], row[6]) < cfg.moving_threshold:
            v_o = 0.0
        rel = row[2] - yaw
        o_half = row[3] * abs(np.cos(rel)) + row[4] * abs(np.sin(rel))
        lengths = cfg.ego_half_length + o_half
        d = (s_o0 + v_o * ts) - (s_e0 + v_e * ts)
        need_ahead = lengths + cfg.front_buffer_m + cfg.front_time_gap_s * v_e
        need_behind = lengths + cfg.rear_buffer_m + cfg.rear_time_gap_s * max(v_o, 0.0)
        ahead_bad = (d >= 0) & (d < need_ahead)
        behind_bad = (d < 0) & (-d < need_behind)
        bad = ahead_bad | behind_bad
        if np.any(bad):
            first = int(np.argmax(bad))                 # earliest conflict
            where = "ahead" if ahead_bad[first] else "behind"
            return False, f"car {where} in target lane (in {ts[first]:.1f} s)"
    return True, "clear"


# ── State machine ────────────────────────────────────────────────────────────

class BehaviorPlanner:
    def __init__(self, config: Optional[BehaviorConfig] = None):
        self.cfg = config or BehaviorConfig()
        self.reset()

    def reset(self):
        self.state = LANE_KEEP
        self.target: Optional[str] = None
        self._t: Optional[float] = None
        self._slow_since: Optional[float] = None
        self._gap_ok_since: Optional[float] = None
        self._cooldown_until = -np.inf
        self._d = 0.0          # distance travelled since the change / abort started
        self._length = 1.0
        self._w0 = 0.0         # progress when an abort started
        self._conflicts = 0    # consecutive gap conflicts while changing
        self.last_reason = ""

    def _progress(self) -> float:
        if self.state == CHANGING:
            return smoothstep(self._d / self._length)
        if self.state == ABORT:
            return self._w0 * (1.0 - smoothstep(self._d / self._length))
        return 0.0

    def _change_length(self, ego_speed: float) -> float:
        cfg = self.cfg
        return min(cfg.change_length_m, max(cfg.min_change_length_m, cfg.change_time_s * ego_speed))

    def _window(self, ego_speed: float) -> float:
        return max(self.cfg.min_window_s, self._change_length(ego_speed) / max(ego_speed, 1.0) + 1.0)

    def _pick_target(self, lanes: dict, ego_xy, ego_speed, obs) -> tuple[Optional[str], str]:
        reason = "no same-direction lane"
        for side in self.cfg.prefer:
            lane = lanes.get(side)
            if lane is None or lane.direction != SAME:
                continue
            ok, why = target_lane_clear(lane, ego_xy, ego_speed, obs,
                                        self._window(ego_speed), self.cfg)
            if ok:
                return side, "clear"
            reason = why
        return None, reason

    def update(self, t: float, ego_xy, ego_speed: float, lanes: dict,
               obstacles: np.ndarray) -> BehaviorOutput:
        """
        One decision step.

        t         : time (s, simulated)
        lanes     : {'ego': Lane, 'left': Lane | None, 'right': Lane | None}
        obstacles : (N, 7) array as in obstacles.as_obstacle_array
        """
        cfg = self.cfg
        obs = np.asarray(obstacles, float).reshape(-1, 7)
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        switch = None
        ego_lane = lanes["ego"]
        lead = lead_in_lane(ego_lane, ego_xy, obs, cfg)
        slow = lead is not None and lead[1] < cfg.slow_ratio * cfg.target_speed
        if slow:
            self._slow_since = t if self._slow_since is None else self._slow_since
        else:
            self._slow_since = None

        if self.state in (CHANGING, ABORT):
            self._d += ego_speed * dt
            target_lane = lanes.get(self.target)
            if self.state == CHANGING:
                if target_lane is None:
                    self._start_abort("target lane ended")
                else:
                    ok, why = target_lane_clear(target_lane, ego_xy, ego_speed, obs,
                                                self._window(ego_speed), cfg)
                    self._conflicts = 0 if ok else self._conflicts + 1
                    if (self._conflicts >= cfg.abort_confirm_ticks
                            and self._progress() < cfg.abort_progress_max):
                        self._start_abort(why)
                    elif self._d >= self._length:
                        switch = self.target
                        self._finish(t, "change complete")
            elif self._d >= self._length:
                self._finish(t, "abort complete")
        else:
            if lead is None:
                self.state = LANE_KEEP
                self._gap_ok_since = None
            elif self.state == LANE_KEEP:
                self.state = FOLLOW
            if self.state in (FOLLOW, PREPARE_CHANGE):
                wants = (slow and t - self._slow_since >= cfg.slow_time_s
                         and t >= self._cooldown_until)
                if not wants:
                    self.state = FOLLOW if lead is not None else LANE_KEEP
                    self._gap_ok_since = None
                else:
                    side, why = self._pick_target(lanes, ego_xy, ego_speed, obs)
                    self.last_reason = why
                    if side is None:
                        self.state = PREPARE_CHANGE if any(
                            lanes.get(s) is not None and lanes[s].direction == SAME
                            for s in cfg.prefer) else FOLLOW
                        self._gap_ok_since = None
                    else:
                        self.state = PREPARE_CHANGE
                        if self._gap_ok_since is None:
                            self._gap_ok_since = t
                        if t - self._gap_ok_since >= cfg.prepare_time_s:
                            self.state, self.target = CHANGING, side
                            self._d, self._length = 0.0, self._change_length(ego_speed)
                            self.last_reason = f"changing {side}"

        progress = self._progress()
        return BehaviorOutput(
            state=self.state, target=self.target if self.state in (CHANGING, ABORT) else None,
            switch_lane=switch, progress=progress, reason=self.last_reason,
            _d=self._d, _length=self._length, _w0=self._w0, _aborting=self.state == ABORT,
        )

    def _start_abort(self, why: str):
        self._conflicts = 0
        self._w0 = self._progress()
        self.state = ABORT
        self._d, self._length = 0.0, self.cfg.abort_length_m
        self.last_reason = f"abort: {why}"

    def _finish(self, t: float, why: str):
        self.state = LANE_KEEP
        self.target = None
        self._d, self._w0 = 0.0, 0.0
        self._cooldown_until = t + self.cfg.cooldown_s
        self._gap_ok_since = None
        self._slow_since = None
        self.last_reason = why
