#!/usr/bin/env python3
"""
mppi_planner.py — Road-Constrained MPPI
========================================
A kinematic-rollout MPPI (first version) failed on turns: most sampled
bicycle-model rollouts go straight, so the importance-weighted update
converged to "mostly straight" even when the road turned.

This version optimises *where on the road* to be rather than *how to move
through free space*: trajectories are parameterised as lateral offsets
from the road-graph centre-line, so road-following is guaranteed by
construction and MPPI only handles lateral positioning (lane centring,
obstacle avoidance).

Algorithm per planning cycle
-----------------------------
  1. Receive road-graph centre-line: N waypoints of (ros_x, ros_y, ros_yaw),
     spaced `wp_step` metres apart and anchored at the ego position.
  2. Shift the warm-start U* by the distance the ego advanced since the
     previous cycle (fractional, via interpolation), so each offset stays
     attached to the same stretch of road.
  3. Sample K lateral-offset perturbation sequences  δU[k] ~ N(0, σ²).
  4. Build K trajectories:
         traj[k][i] = centre_line[i] + (U*[i] + δU[k][i]) × lateral_dir[i]
  5. Score each trajectory (lane deviation, smoothness, offset magnitude,
     obstacle clearance).
  6. Importance weights  w[k] = exp(−(J[k] − min J) / λ).
  7. U* ← U* + α · Σ w[k]·δU[k]     (α = update_step_size)
  8. Return the trajectory built from the updated U*.

Coordinate conventions
----------------------
    ros_x = carla_x,  ros_y = -carla_y,  ros_yaw = -carla_yaw
    Positive lateral offset = LEFT of the centre-line.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class MPPIConfig:
    # Sampling
    num_samples:        int   = 400     # K — number of candidate trajectories
    horizon:            int   = 40      # H — waypoints to optimise over

    # Lateral offset noise (metres)
    lateral_std:        float = 0.4     # σ — perturbation standard deviation
    max_lateral_offset: float = 1.5     # hard clamp on total offset from centre

    # Temperature — lower λ → greedier (sharper weight distribution)
    temperature:        float = 0.5

    # Step size α on the weighted update (1.0 = textbook MPPI)
    update_step_size:   float = 1.0

    # Cost weights
    w_lane:             float = 3.0     # squared lateral deviation from centre-line
    w_smoothness:       float = 2.0     # squared rate of change of lateral offset
    w_deviation:        float = 0.5     # squared offset magnitude (prefer centre)

    # Obstacle avoidance: cost w·((r/d)² − 1) when a trajectory passes within
    # r = min_clearance_m of an obstacle centre (continuous, 0 at d = r).
    w_obstacle:         float = 10.0
    min_clearance_m:    float = 2.8
    # Longitudinal distance is divided by this before computing clearance,
    # so the elongated vehicle footprint is respected: points a few metres
    # before/after the obstacle must also be laterally clear.
    obstacle_long_scale: float = 2.5

    seed:               Optional[int] = None


class MPPIPlanner:
    """
    Usage
    -----
        planner = MPPIPlanner(config)
        trajectory = planner.plan(
            centre_line        = [(ros_x, ros_y, ros_yaw), ...],
            obstacle_positions = [(ros_x, ros_y), ...],
            advance_wps        = distance_moved / wp_step,
        )
    """

    def __init__(self, config: Optional[MPPIConfig] = None):
        self.cfg = config or MPPIConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._U_star = np.zeros(self.cfg.horizon)   # nominal offsets, + = left

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        centre_line:        Sequence[tuple[float, float, float]],
        obstacle_positions: Sequence[tuple[float, float]] = (),
        advance_wps:        float = 0.0,
    ) -> list[tuple[float, float, float]]:
        cfg = self.cfg
        if not centre_line:
            return []
        H = min(cfg.horizon, len(centre_line))
        if H < 2:
            return list(centre_line)
        K = cfg.num_samples

        cl = np.asarray(centre_line[:H], dtype=float)
        cl_x, cl_y, cl_yaw = cl[:, 0], cl[:, 1], cl[:, 2]
        lat_x = -np.sin(cl_yaw)          # left-perpendicular in ROS frame
        lat_y = np.cos(cl_yaw)

        # ── Warm start, shifted by the ego's progress along the road ─────
        U_star = self._shifted_warm_start(H, advance_wps)

        # ── Sample and score ─────────────────────────────────────────────
        delta_U = self._rng.standard_normal((K, H)) * cfg.lateral_std
        U_k = np.clip(U_star[None, :] + delta_U, -cfg.max_lateral_offset, cfg.max_lateral_offset)
        delta_U = U_k - U_star[None, :]   # keep update consistent with the clamp

        traj_x = cl_x[None, :] + U_k * lat_x[None, :]
        traj_y = cl_y[None, :] + U_k * lat_y[None, :]
        costs = self._compute_costs(traj_x, traj_y, cl_x, cl_y, cl_yaw, U_k, obstacle_positions)

        # ── Importance-weighted update ───────────────────────────────────
        beta = costs.min()
        weights = np.exp(-(costs - beta) / cfg.temperature)
        weights /= weights.sum() + 1e-12
        U_star = U_star + cfg.update_step_size * (weights[:, None] * delta_U).sum(axis=0)
        U_star = np.clip(U_star, -cfg.max_lateral_offset, cfg.max_lateral_offset)

        self._U_star = np.zeros(cfg.horizon)
        self._U_star[:H] = U_star

        # ── Trajectory from the optimised U* (heading from geometry) ─────
        xs = cl_x + U_star * lat_x
        ys = cl_y + U_star * lat_y
        yaws = _headings(xs, ys, cl_yaw)
        return [(float(x), float(y), float(a)) for x, y, a in zip(xs, ys, yaws)]

    def reset(self):
        self._U_star = np.zeros(self.cfg.horizon)

    @property
    def nominal_offsets(self) -> np.ndarray:
        return self._U_star.copy()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _shifted_warm_start(self, H: int, advance_wps: float) -> np.ndarray:
        prev = self._U_star
        if advance_wps <= 1e-6:
            return prev[:H].copy()
        idx = np.arange(H) + advance_wps
        # Points that fall beyond the previous horizon decay to centre (0).
        return np.interp(idx, np.arange(len(prev)), prev, right=0.0)

    def _compute_costs(
        self,
        traj_x: np.ndarray, traj_y: np.ndarray,       # (K, H)
        cl_x: np.ndarray, cl_y: np.ndarray,           # (H,)
        cl_yaw: np.ndarray,                           # (H,)
        U_k: np.ndarray,                              # (K, H)
        obstacle_positions: Sequence[tuple[float, float]],
    ) -> np.ndarray:
        cfg = self.cfg
        cte = (traj_x - cl_x[None, :]) ** 2 + (traj_y - cl_y[None, :]) ** 2
        costs = cfg.w_lane * cte.mean(axis=1)
        costs += cfg.w_smoothness * (np.diff(U_k, axis=1) ** 2).mean(axis=1)
        costs += cfg.w_deviation * (U_k ** 2).mean(axis=1)

        if len(obstacle_positions):
            obs = np.asarray(obstacle_positions, dtype=float)               # (N, 2)
            d_eff = footprint_distance(traj_x, traj_y, cl_yaw, obs,
                                       cfg.obstacle_long_scale)            # (K, H, N)
            # Summed over every point near an obstacle (not just the closest
            # one) so the whole stretch alongside it is pushed out, and
            # continuous at d = r so the optimum doesn't sit on the boundary.
            ratio = cfg.min_clearance_m / np.maximum(d_eff, 1e-3)
            costs += cfg.w_obstacle * np.maximum(ratio ** 2 - 1.0, 0.0).sum(axis=(1, 2))
        return costs


def footprint_distance(xs, ys, yaws, obs, long_scale):
    """
    Elliptical distance from path points to obstacles, shape (..., H, N).

    Splits the offset into along-path and lateral parts (using the path
    heading) and divides the along-path part by `long_scale`.
    """
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    c, s = np.cos(yaws), np.sin(yaws)
    dx = xs[..., None] - obs[:, 0]
    dy = ys[..., None] - obs[:, 1]
    along = dx * c[..., None] + dy * s[..., None]
    lat = -dx * s[..., None] + dy * c[..., None]
    return np.sqrt((along / long_scale) ** 2 + lat ** 2)


def _headings(xs: np.ndarray, ys: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Per-point heading from finite differences of the path itself."""
    if len(xs) < 2:
        return fallback.copy()
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    yaws = np.arctan2(dy, dx)
    degenerate = np.hypot(dx, dy) < 1e-6
    yaws[degenerate] = fallback[degenerate]
    return yaws
