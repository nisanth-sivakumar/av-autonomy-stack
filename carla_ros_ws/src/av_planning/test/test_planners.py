import math

import numpy as np

from av_planning.lattice_planner import LatticePlanner, LatticeConfig
from av_planning.mppi_planner import MPPIPlanner, MPPIConfig

STEP = 3.0


def straight(n=40):
    return [(i * STEP + STEP, 0.0, 0.0) for i in range(n)]


def left_curve(n=40, radius=30.0):
    pts = []
    for i in range(n):
        s = (i + 1) * STEP
        a = s / radius
        pts.append((radius * math.sin(a), radius * (1 - math.cos(a)), a))
    return pts


def lateral(points, centre):
    """Signed lateral offset of each point from the matching centre point (+ = left)."""
    return np.array([
        (x - cx) * -math.sin(cyaw) + (y - cy) * math.cos(cyaw)
        for (x, y, _), (cx, cy, cyaw) in zip(points, centre)
    ])


# ── Lattice ──────────────────────────────────────────────────────────────────

def test_lattice_follows_centre_without_obstacles():
    cl = straight()
    traj = LatticePlanner().plan(cl, [], 0.0)
    assert len(traj) == len(cl)
    assert np.abs(lateral(traj, cl)).max() < 1e-9


def test_lattice_blends_from_current_offset():
    cl = straight()
    traj = LatticePlanner().plan(cl, [], current_offset_m=1.0)
    lat = lateral(traj, cl)
    assert abs(lat[0] - 1.0) < 0.15         # starts near the ego offset (3 m in)
    assert np.all(np.diff(lat) <= 1e-9)     # monotone return toward centre
    assert abs(lat[-1]) < 1e-9


def test_lattice_avoids_obstacle_beside_centre():
    cl = straight()
    obstacle = (30.0, 0.8)                     # slightly left of centre
    traj = LatticePlanner().plan(cl, [obstacle], 0.0)
    lat = lateral(traj, cl)
    assert lat[-1] < -0.5                      # shifted right, away from it
    d = min(math.hypot(x - obstacle[0], y - obstacle[1]) for x, y, _ in traj)
    assert d >= LatticeConfig().min_clearance_m


def test_lattice_fully_blocked_returns_max_clearance_candidate():
    cl = straight()
    obstacles = [(30.0, y) for y in np.linspace(-4, 4, 9)]
    p = LatticePlanner()
    traj = p.plan(cl, obstacles, 0.0)
    assert not any(c.feasible for c in p.last_candidates)
    best = max(p.last_candidates, key=lambda c: c.min_clearance)
    assert traj == best.waypoints


def test_lattice_infeasible_ramp_prefers_side_over_centre():
    # Obstacle partly in lane, too close for the blend to reach full offset:
    # every candidate is infeasible, but the fallback must not be the
    # centre-line (which drives straight at it).
    cl = straight()
    traj = LatticePlanner().plan(cl, [(9.0, 1.0)], 0.0)
    assert lateral(traj, cl)[-1] < -1.0


def test_lattice_headings_follow_geometry_on_curve():
    cl = left_curve()
    traj = LatticePlanner().plan(cl, [], 0.0)
    for (_, _, yaw), (_, _, cyaw) in zip(traj[1:-1], cl[1:-1]):
        assert abs(math.remainder(yaw - cyaw, 2 * math.pi)) < 0.05


# ── MPPI ────────────────────────────────────────────────────────────────────

def test_mppi_config_accepts_update_step_size():
    MPPIConfig(update_step_size=0.5)   # regression: planner_node passes this


def test_mppi_stays_near_centre_without_obstacles():
    cl = straight()
    p = MPPIPlanner(MPPIConfig(seed=0, lateral_std=0.2, temperature=0.3))
    for _ in range(20):
        traj = p.plan(cl, [], advance_wps=0.1)
    assert np.abs(lateral(traj, cl)).max() < 0.3


def test_mppi_follows_curve():
    cl = left_curve()
    p = MPPIPlanner(MPPIConfig(seed=0))
    traj = p.plan(cl, [])
    assert np.abs(lateral(traj, cl)).max() < 1.5
    assert traj[-1][1] > 10.0          # actually turned left


def test_mppi_moves_away_from_obstacle():
    cl = straight()
    obstacle = (30.0, 0.3)
    p = MPPIPlanner(MPPIConfig(seed=0, lateral_std=0.4, temperature=0.3,
                               max_lateral_offset=1.8, min_clearance_m=1.5))
    for _ in range(40):
        traj = p.plan(cl, [obstacle])
    d = min(math.hypot(x - obstacle[0], y - obstacle[1]) for x, y, _ in traj)
    assert d > 1.0, d


def test_mppi_warm_start_shift_tracks_progress():
    p = MPPIPlanner(MPPIConfig(horizon=10))
    p._U_star = np.arange(10, dtype=float)
    shifted = p._shifted_warm_start(10, advance_wps=1.5)
    assert np.allclose(shifted[:8], np.arange(10)[:8] + 1.5)
    assert shifted[-1] == 0.0
    assert np.allclose(p._shifted_warm_start(10, 0.0), np.arange(10))


def test_mppi_handles_short_and_empty_centre_line():
    p = MPPIPlanner()
    assert p.plan([], []) == []
    assert p.plan([(1.0, 0.0, 0.0)], []) == [(1.0, 0.0, 0.0)]


def test_mppi_clears_whole_stretch_alongside_obstacle():
    # Footprint-shaped cost: points a few metres before/after the obstacle
    # must be laterally clear too, not only the closest waypoint.
    cl = straight(60)
    obstacle = (60.0, 1.0)
    p = MPPIPlanner(MPPIConfig(seed=1, lateral_std=0.2, temperature=0.3, update_step_size=0.5,
                               w_lane=3.0, w_smoothness=5.0, w_deviation=0.2))
    for _ in range(60):
        traj = p.plan(cl, [obstacle])
    lat = lateral(traj, cl)
    near = [i for i, (x, _, _) in enumerate(cl) if abs(x - obstacle[0]) <= 4.5]
    assert all(obstacle[1] - lat[i] >= 2.3 for i in near), lat[near]


def test_lattice_does_not_chase_vehicle_tracking_error():
    # Regression (CARLA junction turn): the car sits 0.8 m inside the
    # previous plan because Pure Pursuit cuts the corner. The next plan must
    # continue from the previous plan (≈ centre), not restart at the car.
    cl = left_curve(radius=12.0)
    p = LatticePlanner()
    first = p.plan(cl, [], current_offset_m=0.0)
    assert abs(lateral(first, cl)[0]) < 1e-6
    # advance one waypoint along the curve; the car is 0.8 m inside (left)
    second = p.plan(cl[1:], [], current_offset_m=0.8)
    assert abs(lateral(second, cl[1:])[0]) < 0.1
