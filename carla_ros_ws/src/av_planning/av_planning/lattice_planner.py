#!/usr/bin/env python3
"""
lattice_planner.py — Phase 2b
==============================
Samples a set of lateral offset candidate trajectories around a centre-line
produced by the CARLA road graph, scores each candidate, and returns the
lowest-cost driveable trajectory.

Algorithm overview
------------------
1. Generate a centre-line from the CARLA road graph  (reuses road graph walk)
2. For each centre-line waypoint, sample N lateral offsets → candidate paths
3. Score each candidate path with a weighted cost function:
      J = w_dev   * lane_deviation²
        + w_curv  * curvature²
        + w_smooth* steering_jerk²
        + w_clear * (1 / clearance_to_obstacles)   ← Phase 3 hook
4. Return the minimum-cost path

Coordinate conventions
----------------------
    ros_x = carla_x,  ros_y = -carla_y,  ros_yaw = -carla_yaw
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

Waypoint = tuple[float, float, float]   # (ros_x, ros_y, ros_yaw)


@dataclass
class LatticeConfig:
    # Lateral sampling
    num_offsets:      int   = 7        # number of lateral offsets to sample
    max_offset_m:     float = 1.8      # ± metres from centre-line
    # Cost weights
    w_deviation:      float = 1.0      # penalise distance from centre-line
    w_curvature:      float = 0.5      # penalise sharp turns
    w_smoothness:     float = 0.3      # penalise steering jerk
    # Collision hook (Phase 3: populate obstacle_positions to activate)
    w_clearance:      float = 2.0      # weight for obstacle proximity cost
    min_clearance_m:  float = 0.5      # below this → path is infeasible


@dataclass
class CandidatePath:
    waypoints:  list[Waypoint] = field(default_factory=list)
    offset_m:   float          = 0.0
    cost:       float          = float("inf")
    feasible:   bool           = True


# ---------------------------------------------------------------------------
# Lattice planner
# ---------------------------------------------------------------------------

class LatticePlanner:
    """
    Lateral lattice planner.

    Usage
    -----
    planner = LatticePlanner(config)
    trajectory = planner.plan(centre_line, obstacle_positions=[])

    centre_line         : list of (ros_x, ros_y, ros_yaw) from road graph
    obstacle_positions  : list of (ros_x, ros_y) — Phase 3 hook, leave [] for now
    """

    def __init__(self, config: Optional[LatticeConfig] = None):
        self.cfg = config or LatticeConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        centre_line:        list[Waypoint],
        obstacle_positions: list[tuple[float, float]] = [],
    ) -> list[Waypoint]:
        """
        Generate and score candidate trajectories.
        Returns the lowest-cost feasible trajectory.
        Falls back to centre-line if all candidates are infeasible.
        """
        if not centre_line:
            return []

        offsets    = self._sample_offsets()
        candidates = [
            self._build_candidate(centre_line, offset)
            for offset in offsets
        ]

        for c in candidates:
            c.cost     = self._score(c, centre_line, obstacle_positions)
            c.feasible = self._is_feasible(c, obstacle_positions)

        feasible = [c for c in candidates if c.feasible]
        if not feasible:
            # All paths blocked — return centre-line as emergency fallback
            return centre_line

        best = min(feasible, key=lambda c: c.cost)
        return best.waypoints

    # ------------------------------------------------------------------
    # Lateral offset sampling
    # ------------------------------------------------------------------

    def _sample_offsets(self) -> list[float]:
        """
        Return evenly-spaced lateral offsets in [-max_offset, +max_offset].
        Always includes 0.0 (centre-line).
        """
        n   = self.cfg.num_offsets
        max = self.cfg.max_offset_m
        if n == 1:
            return [0.0]
        return [max * (2 * i / (n - 1) - 1) for i in range(n)]

    # ------------------------------------------------------------------
    # Candidate path construction
    # ------------------------------------------------------------------

    def _build_candidate(
        self,
        centre_line: list[Waypoint],
        offset_m:    float,
    ) -> CandidatePath:
        """
        Shift every centre-line waypoint laterally by offset_m.

        Positive offset → left of centre-line  (in ROS frame)
        Negative offset → right of centre-line

        The lateral direction at each waypoint is the unit vector
        perpendicular to the road heading (yaw).
        """
        shifted = []
        for (x, y, yaw) in centre_line:
            # Left-perpendicular in ROS frame: rotate heading 90° CCW
            lat_x = -math.sin(yaw)
            lat_y =  math.cos(yaw)
            sx = x + offset_m * lat_x
            sy = y + offset_m * lat_y
            shifted.append((sx, sy, yaw))

        return CandidatePath(waypoints=shifted, offset_m=offset_m)

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def _score(
        self,
        candidate:          CandidatePath,
        centre_line:        list[Waypoint],
        obstacle_positions: list[tuple[float, float]],
    ) -> float:
        cfg = self.cfg
        wps = candidate.waypoints
        n   = len(wps)
        if n == 0:
            return float("inf")

        # ── Lane deviation cost ───────────────────────────────────────
        # Mean squared distance from centre-line
        dev_cost = 0.0
        for i, (cx, cy, _) in enumerate(centre_line[:n]):
            wx, wy, _ = wps[i]
            dev_cost += (wx - cx) ** 2 + (wy - cy) ** 2
        dev_cost /= n

        # ── Curvature cost ────────────────────────────────────────────
        # Approximate curvature via second derivative of path
        curv_cost = 0.0
        if n >= 3:
            for i in range(1, n - 1):
                x0, y0, _ = wps[i - 1]
                x1, y1, _ = wps[i]
                x2, y2, _ = wps[i + 1]
                dx1, dy1  = x1 - x0, y1 - y0
                dx2, dy2  = x2 - x1, y2 - y1
                # Cross product magnitude ≈ curvature × ds²
                cross     = dx1 * dy2 - dy1 * dx2
                ds        = math.hypot(dx1, dy1) + 1e-6
                curv_cost += (cross / ds ** 2) ** 2
            curv_cost /= (n - 2)

        # ── Smoothness cost (heading change = steering jerk proxy) ────
        smooth_cost = 0.0
        if n >= 3:
            for i in range(1, n - 1):
                _, _, yaw0 = wps[i - 1]
                _, _, yaw1 = wps[i]
                _, _, yaw2 = wps[i + 1]
                d1 = self._angle_diff(yaw1, yaw0)
                d2 = self._angle_diff(yaw2, yaw1)
                smooth_cost += (d2 - d1) ** 2
            smooth_cost /= (n - 2)

        # ── Clearance cost (Phase 3 hook) ─────────────────────────────
        clear_cost = 0.0
        if obstacle_positions:
            min_dist = self._min_clearance(wps, obstacle_positions)
            if min_dist > 1e-3:
                clear_cost = 1.0 / min_dist

        total = (
            cfg.w_deviation  * dev_cost   +
            cfg.w_curvature  * curv_cost  +
            cfg.w_smoothness * smooth_cost +
            cfg.w_clearance  * clear_cost
        )
        return total

    # ------------------------------------------------------------------
    # Feasibility check
    # ------------------------------------------------------------------

    def _is_feasible(
        self,
        candidate:          CandidatePath,
        obstacle_positions: list[tuple[float, float]],
    ) -> bool:
        """
        A candidate is infeasible if it passes within min_clearance_m
        of any known obstacle.  With no obstacles, all paths are feasible.
        """
        if not obstacle_positions:
            return True
        clearance = self._min_clearance(candidate.waypoints, obstacle_positions)
        return clearance >= self.cfg.min_clearance_m

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _min_clearance(
        self,
        waypoints:          list[Waypoint],
        obstacle_positions: list[tuple[float, float]],
    ) -> float:
        min_dist = float("inf")
        for (wx, wy, _) in waypoints:
            for (ox, oy) in obstacle_positions:
                d = math.hypot(wx - ox, wy - oy)
                if d < min_dist:
                    min_dist = d
        return min_dist

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Signed angular difference a - b, wrapped to [-π, π]."""
        d = a - b
        while d >  math.pi: d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        return d