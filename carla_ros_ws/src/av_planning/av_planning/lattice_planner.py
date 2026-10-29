#!/usr/bin/env python3
"""
lattice_planner.py — Lateral lattice planner
=============================================
Samples lateral-offset candidates around the road-graph centre-line,
scores them, and returns the lowest-cost feasible one.

Candidate construction
----------------------
Each candidate starts at an offset d0 and blends to a target offset d_t
over `transition_length_m` with a smoothstep (3s² − 2s³), then holds d_t.
This gives smooth, lane-change-shaped manoeuvres instead of a step to a
constant offset, and makes the curvature / smoothness terms actually
discriminate between candidates.

d0 is the *previous plan's* offset at the start of the new centre-line
(plan continuity), not the vehicle's measured offset.  Starting from the
vehicle's offset makes the plan chase the controller's tracking error:
in a turn Pure Pursuit cuts slightly inside, the next plan starts inside,
the car cuts inside of that, and so on.  In a tight CARLA junction turn
this walked the car onto the inner kerb.  The vehicle's offset is only
used for the very first plan (or when the previous plan doesn't cover
the new start point).

Cost
----
    J = w_dev    · mean(offset²)
      + w_curv   · mean(κ²)
      + w_smooth · mean((Δκ)²)
      + w_clear  · Σ_obstacles max(0, comfort_clearance − d_min)²
      + w_consistency · (d_t − previous d_t)²       (hysteresis)

Clearance is measured to each obstacle's box at its predicted position
when the ego gets there (obstacles.box_clearance): the lateral gap from
the ego's centreline to the box edge wherever the ego's footprint
overlaps the obstacle lengthwise.  A candidate is infeasible below
`min_clearance_m`.  If every candidate is
infeasible the planner returns the one with the most clearance (the
centre-line would be the path most certainly blocked); the controller's
obstacle-aware speed law stops the vehicle if that is still not enough.

Coordinate conventions
----------------------
    ros_x = carla_x,  ros_y = -carla_y,  ros_yaw = -carla_yaw
    Positive lateral offset = LEFT of the centre-line.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from av_planning.obstacles import as_obstacle_array, avoidable, box_clearance, path_times

Waypoint = tuple[float, float, float]   # (ros_x, ros_y, ros_yaw)


@dataclass
class LatticeConfig:
    # Lateral sampling
    num_offsets:          int   = 7        # number of target offsets to sample
    max_offset_m:         float = 1.8      # ± metres from centre-line
    transition_length_m:  float = 15.0     # distance to blend from d0 to d_t
    wp_step_m:            float = 3.0      # spacing of centre-line waypoints

    # Cost weights
    w_deviation:          float = 1.0
    w_curvature:          float = 0.5
    w_smoothness:         float = 0.3
    w_clearance:          float = 2.0
    w_consistency:        float = 0.5

    # Obstacles: clearance from the ego centreline to the obstacle box edge
    # (footprints touch below the ego half-width ~0.93 m; the controller
    # treats < 1.35 m from its path as blocking, so plan with more margin)
    min_clearance_m:      float = 1.75     # below → infeasible
    comfort_clearance_m:  float = 2.35     # below → clearance cost
    ego_half_length:      float = 2.35     # Tesla Model 3: 4.7 m long
    # Obstacles that can't be passed with at least this clearance even at
    # max_offset_m are left to the controller (it stops; = its corridor)
    pass_clearance_m:     float = 1.35
    # Must also be clear this far before/after the obstacle (early, smooth swerve)
    obstacle_long_margin_m: float = 3.0


@dataclass
class CandidatePath:
    waypoints:  list[Waypoint] = field(default_factory=list)
    offset_m:   float          = 0.0      # target offset d_t
    offsets:    list[float]    = field(default_factory=list)  # per-waypoint offset
    cost:       float          = float("inf")
    feasible:   bool           = True
    min_clearance: float       = float("inf")


class LatticePlanner:
    """
    Usage
    -----
        planner = LatticePlanner(config)
        trajectory = planner.plan(centre_line, obstacle_positions, current_offset_m)
    """

    def __init__(self, config: Optional[LatticeConfig] = None):
        self.cfg = config or LatticeConfig()
        self._prev_offset: float = 0.0
        self._last_plan_xy: Optional[np.ndarray] = None
        self.last_candidates: list[CandidatePath] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        centre_line:        Sequence[Waypoint],
        obstacle_positions: Sequence = (),
        current_offset_m:   float = 0.0,
        ego_xy:             Optional[tuple[float, float]] = None,
        ego_speed:          float = 0.0,
    ) -> list[Waypoint]:
        if not centre_line:
            return []
        cl = np.asarray(centre_line, dtype=float)
        # lead cars (same direction) are followed by the controller, not avoided
        obs = avoidable(as_obstacle_array(obstacle_positions), cl[:, :2], cl[:, 2],
                        max_offset=self.cfg.max_offset_m,
                        pass_clearance=self.cfg.pass_clearance_m)
        times = path_times(cl[:, 0], cl[:, 1], ego_xy, ego_speed) if len(obs) else None

        d0 = self._continuity_offset(cl)
        if d0 is None:
            d0 = current_offset_m
        candidates = [
            self._build_candidate(cl, d0, d_t)
            for d_t in self._sample_offsets()
        ]
        for c in candidates:
            pts = np.asarray([(x, y) for x, y, _ in c.waypoints])
            min_d = _min_clearance_per_obstacle(pts, cl[:, 2], obs, self.cfg.ego_half_length,
                                                times, self.cfg.obstacle_long_margin_m)
            c.feasible = bool(min_d.size == 0 or min_d.min() >= self.cfg.min_clearance_m)
            c.min_clearance = float(min_d.min()) if min_d.size else float("inf")
            c.cost = self._score(c, pts, min_d)
        self.last_candidates = candidates

        feasible = [c for c in candidates if c.feasible]
        if not feasible:
            best = max(candidates, key=lambda c: c.min_clearance)
        else:
            best = min(feasible, key=lambda c: c.cost)
        self._prev_offset = best.offset_m
        self._last_plan_xy = np.asarray([(x, y) for x, y, _ in best.waypoints])
        return best.waypoints

    def reset(self):
        self._prev_offset = 0.0
        self._last_plan_xy = None

    def _continuity_offset(self, cl: np.ndarray) -> Optional[float]:
        """Lateral offset of the previous plan at the new centre-line's first point."""
        prev = self._last_plan_xy
        if prev is None or len(prev) < 2:
            return None
        x0, y0, yaw0 = cl[0]
        a, b = prev[:-1], prev[1:]
        ab = b - a
        t = np.clip(((np.array([x0, y0]) - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9),
                    0.0, 1.0)
        proj = a + t[:, None] * ab
        # the previous plan's point *beside* cl[0]: smallest along-track distance
        along = (proj[:, 0] - x0) * np.cos(yaw0) + (proj[:, 1] - y0) * np.sin(yaw0)
        lat = -(proj[:, 0] - x0) * np.sin(yaw0) + (proj[:, 1] - y0) * np.cos(yaw0)
        i = int(np.argmin(np.abs(along) + 0.1 * np.abs(lat)))
        if abs(along[i]) > self.cfg.wp_step_m or abs(lat[i]) > self.cfg.max_offset_m + 0.5:
            return None      # previous plan doesn't reach / match the new start
        return float(lat[i])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample_offsets(self) -> list[float]:
        """Evenly spaced in [-max, +max]; includes 0 when num_offsets is odd."""
        n, m = self.cfg.num_offsets, self.cfg.max_offset_m
        if n <= 1:
            return [0.0]
        return [m * (2 * i / (n - 1) - 1) for i in range(n)]

    def offset_profile(self, n: int, d0: float, d_t: float) -> np.ndarray:
        """Lateral offset at each of n waypoints: smoothstep from d0 to d_t."""
        s_along = (np.arange(n) + 1) * self.cfg.wp_step_m
        u = np.clip(s_along / max(self.cfg.transition_length_m, 1e-6), 0.0, 1.0)
        blend = 3 * u ** 2 - 2 * u ** 3
        return d0 + (d_t - d0) * blend

    def _build_candidate(self, cl: np.ndarray, d0: float, d_t: float) -> CandidatePath:
        offs = self.offset_profile(len(cl), d0, d_t)
        xs = cl[:, 0] - np.sin(cl[:, 2]) * offs
        ys = cl[:, 1] + np.cos(cl[:, 2]) * offs
        yaws = _headings(xs, ys, cl[:, 2])
        wps = [(float(x), float(y), float(a)) for x, y, a in zip(xs, ys, yaws)]
        return CandidatePath(waypoints=wps, offset_m=d_t, offsets=offs.tolist())

    def _score(self, c: CandidatePath, pts: np.ndarray, min_d: np.ndarray) -> float:
        cfg = self.cfg
        n = len(pts)
        if n == 0:
            return float("inf")
        yaws = np.asarray([w[2] for w in c.waypoints])

        # Deviation: mean squared lateral offset = mean squared distance
        # to the centre-line (points are shifted along the normal).
        dev = float(np.mean(np.square(c.offsets)))

        curv = smooth = 0.0
        if n >= 3:
            ds = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])) + 1e-6
            dyaw = np.angle(np.exp(1j * np.diff(yaws)))       # wrapped
            kappa = dyaw / ds
            curv = float(np.mean(kappa ** 2))
            smooth = float(np.mean(np.diff(kappa) ** 2)) if len(kappa) >= 2 else 0.0

        clear = float(np.sum(np.maximum(0.0, cfg.comfort_clearance_m - min_d) ** 2)) \
            if min_d.size else 0.0
        consistency = (c.offset_m - self._prev_offset) ** 2

        return (
            cfg.w_deviation * dev
            + cfg.w_curvature * curv
            + cfg.w_smoothness * smooth
            + cfg.w_clearance * clear
            + cfg.w_consistency * consistency
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _min_clearance_per_obstacle(pts: np.ndarray, yaws: np.ndarray, obs: np.ndarray,
                                ego_half_length: float, times, long_margin: float) -> np.ndarray:
    """Minimum ego-to-box clearance along the path for each obstacle, shape (N,)."""
    if obs.size == 0 or pts.size == 0:
        return np.empty(0)
    return box_clearance(pts[:, 0], pts[:, 1], yaws, obs, ego_half_length, times,
                         long_margin=long_margin).min(axis=0)


def _headings(xs: np.ndarray, ys: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if len(xs) < 2:
        return np.asarray(fallback, dtype=float).copy()
    dx, dy = np.gradient(xs), np.gradient(ys)
    yaws = np.arctan2(dy, dx)
    bad = np.hypot(dx, dy) < 1e-6
    yaws[bad] = np.asarray(fallback)[bad]
    return yaws
