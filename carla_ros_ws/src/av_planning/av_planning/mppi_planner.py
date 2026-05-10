#!/usr/bin/env python3
"""
mppi_planner.py — Phase 2b  (Road-Constrained MPPI)
=====================================================
Algorithm
---------
The previous kinematic-rollout MPPI failed on turns because the bicycle
model rollout is unconstrained — most sampled trajectories go straight,
so the importance-weighted update converges to "mostly straight" even
when the road turns.

This implementation uses road-constrained MPPI:

  Instead of rolling out through free space, trajectories are parameterised
  as lateral offsets from the road-graph centre-line provided by the planner.
  MPPI optimises *where on the road* to be, not *how to navigate space*.

  Road-following is guaranteed by construction — the centre-line handles
  all turns. MPPI handles lateral positioning (lane centering, obstacle
  avoidance in Phase 3).

Algorithm per planning cycle
-----------------------------
  1. Receive road-graph centre-line: N waypoints of (ros_x, ros_y, ros_yaw)
  2. Sample K lateral-offset perturbation sequences  δU[k] ~ N(0, σ²)
  3. Build K trajectories by shifting each centre-line point laterally:
         traj[k][i] = centre_line[i]  +  (U*[i] + δU[k][i]) × lateral_dir[i]
  4. Score each trajectory with a cost function
  5. Compute importance weights:  w[k] = exp(−J[k] / λ)
  6. Update nominal offset sequence: U* ← U* + Σ w[k]⋅δU[k]
  7. Return the trajectory built from the updated U*

Advantages over kinematic MPPI
--------------------------------
- Turns are handled automatically by the centre-line
- Cost comparisons are valid (both trajectory and reference at same arc-length)
- Optimisation is 1-D (lateral offsets) not 2-D (steer + accel)
- Warm-start is meaningful across cycles (lateral preference persists)
- No vehicle dynamics model mismatch

Resume framing
--------------
"Road-constrained MPPI: applied importance-sampling trajectory optimisation
over a structured lateral-offset parameterisation of road-graph trajectories,
guaranteeing driveable paths while optimising lane position and obstacle
clearance."

Coordinate conventions
----------------------
    ros_x = carla_x,  ros_y = -carla_y,  ros_yaw = -carla_yaw
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MPPIConfig:
    # Sampling
    num_samples:        int   = 400     # K — number of candidate trajectories
    horizon:            int   = 40      # H — waypoints to optimise over
                                        # (matches road-graph waypoints, not time)

    # Lateral offset noise (metres)
    lateral_std:        float = 0.4     # σ — perturbation standard deviation
    max_lateral_offset: float = 1.5     # hard clamp on total offset from centre

    # Temperature — lower λ → greedier (sharper weight distribution)
    temperature:        float = 0.5

    # Cost weights
    w_lane:             float = 3.0     # squared lateral deviation from centre-line
    w_smoothness:       float = 2.0     # squared rate of change of lateral offset
    w_deviation:        float = 0.5     # squared offset magnitude (prefer centre)

    # Obstacle avoidance (Phase 3 hook — leave [] until perception is added)
    w_obstacle:         float = 10.0
    min_clearance_m:    float = 1.5


# ---------------------------------------------------------------------------
# Road-constrained MPPI planner
# ---------------------------------------------------------------------------

class MPPIPlanner:
    """
    Road-constrained MPPI: optimises lateral offsets along a road-graph
    centre-line.  All candidate trajectories follow the road by construction.

    Usage
    -----
    planner = MPPIPlanner(config)
    trajectory = planner.plan(
        centre_line        = [(ros_x, ros_y, ros_yaw), ...],   # from road graph
        obstacle_positions = [],                                 # Phase 3
    )
    """

    def __init__(self, config: Optional[MPPIConfig] = None):
        self.cfg = config or MPPIConfig()
        # Nominal lateral offset sequence, shape (H,)
        # Positive = left of centre-line, negative = right
        self._U_star = np.zeros(self.cfg.horizon)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        centre_line:        list[tuple[float, float, float]],
        obstacle_positions: list[tuple[float, float]] = [],
    ) -> list[tuple[float, float, float]]:
        """
        Run one MPPI optimisation step over lateral offsets.

        Parameters
        ----------
        centre_line         : road-graph reference, list of (ros_x, ros_y, ros_yaw)
        obstacle_positions  : list of (ros_x, ros_y) — leave [] until Phase 3

        Returns
        -------
        Planned trajectory as list of (ros_x, ros_y, ros_yaw)
        """
        cfg = self.cfg

        if not centre_line:
            return []

        H = min(cfg.horizon, len(centre_line))
        if H < 2:
            return list(centre_line)

        K = cfg.num_samples

        # ── Pre-compute centre-line arrays ────────────────────────────
        cl_x   = np.array([wp[0] for wp in centre_line[:H]])
        cl_y   = np.array([wp[1] for wp in centre_line[:H]])
        cl_yaw = np.array([wp[2] for wp in centre_line[:H]])

        # Lateral direction at each waypoint (left-perpendicular to heading)
        lat_x = -np.sin(cl_yaw)   # ros frame: left = (-sin yaw, cos yaw)
        lat_y =  np.cos(cl_yaw)

        # ── Trim / extend warm-start to H ─────────────────────────────
        U_star = np.zeros(H)
        copy_len = min(H, len(self._U_star))
        U_star[:copy_len] = self._U_star[:copy_len]

        # ── Sample K perturbation sequences ──────────────────────────
        delta_U = np.random.randn(K, H) * cfg.lateral_std

        # ── Perturbed offset sequences  (K, H) ───────────────────────
        U_k = np.clip(
            U_star[None, :] + delta_U,
            -cfg.max_lateral_offset,
             cfg.max_lateral_offset,
        )

        # ── Build K trajectories by lateral shifting ──────────────────
        # traj_x, traj_y: (K, H)
        traj_x = cl_x[None, :] + U_k * lat_x[None, :]
        traj_y = cl_y[None, :] + U_k * lat_y[None, :]

        # ── Score each candidate trajectory ───────────────────────────
        costs = self._compute_costs(
            traj_x, traj_y, cl_x, cl_y, U_k, obstacle_positions
        )

        # ── Importance weights ────────────────────────────────────────
        beta    = costs.min()
        weights = np.exp(-(costs - beta) / cfg.temperature)
        weights /= weights.sum() + 1e-8

        # ── Update nominal offset sequence ────────────────────────────
        update = (weights[:, None] * delta_U).sum(axis=0)
        U_star += update
        U_star  = np.clip(U_star, -cfg.max_lateral_offset, cfg.max_lateral_offset)

        # ── Shift warm-start (receding horizon) ───────────────────────
        self._U_star = np.roll(U_star, -1)
        self._U_star[-1] = 0.0

        # ── Extract planned trajectory from updated U* ─────────────────
        trajectory = []
        for i in range(H):
            x = float(cl_x[i] + self._U_star[i] * lat_x[i])
            y = float(cl_y[i] + self._U_star[i] * lat_y[i])
            trajectory.append((x, y, float(cl_yaw[i])))

        return trajectory

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def _compute_costs(
        self,
        traj_x:             np.ndarray,               # (K, H)
        traj_y:             np.ndarray,               # (K, H)
        cl_x:               np.ndarray,               # (H,)
        cl_y:               np.ndarray,               # (H,)
        U_k:                np.ndarray,               # (K, H) — offset sequences
        obstacle_positions: list[tuple[float, float]],
    ) -> np.ndarray:
        """Returns cost array of shape (K,)."""
        cfg   = self.cfg
        costs = np.zeros(U_k.shape[0])

        # ── Lane deviation: squared distance from centre-line ─────────
        # Since traj = cl + U*lat, deviation IS U*|lat|=U (lat is unit vector)
        # So this is equivalent to U_k**2 but computed geometrically for
        # correctness when obstacles shift the trajectory further.
        cte    = (traj_x - cl_x[None, :]) ** 2 + (traj_y - cl_y[None, :]) ** 2
        costs += cfg.w_lane * cte.mean(axis=1)

        # ── Smoothness: penalise rapid changes in lateral offset ───────
        # This discourages zig-zagging between waypoints.
        if U_k.shape[1] >= 2:
            delta  = np.diff(U_k, axis=1)           # (K, H-1)
            costs += cfg.w_smoothness * (delta ** 2).mean(axis=1)

        # ── Deviation: prefer staying near centre-line ─────────────────
        costs += cfg.w_deviation * (U_k ** 2).mean(axis=1)

        # ── Obstacle clearance (Phase 3 hook) ──────────────────────────
        if obstacle_positions:
            obs_arr  = np.array(obstacle_positions)          # (N, 2)
            traj_xy  = np.stack([traj_x, traj_y], axis=2)   # (K, H, 2)
            diff     = traj_xy[:, :, None, :] - obs_arr[None, None, :, :]
            dist     = np.linalg.norm(diff, axis=3)          # (K, H, N)
            min_dist = dist.min(axis=(1, 2))                  # (K,)
            obs_cost = np.where(
                min_dist < cfg.min_clearance_m,
                cfg.w_obstacle * (cfg.min_clearance_m / (min_dist + 1e-3)) ** 2,
                0.0,
            )
            costs += obs_cost

        return costs

    # ------------------------------------------------------------------
    # Reset (e.g. on new goal or large position jump)
    # ------------------------------------------------------------------

    def reset(self):
        self._U_star = np.zeros(self.cfg.horizon)