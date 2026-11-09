import math

import pytest

from av_control.pure_pursuit import (
    adaptive_lookahead,
    effective_gap,
    path_curvature_ahead,
    path_length_ahead,
    prune_passed,
    find_lookahead,
    pure_pursuit_steer,
    obstacle_along_path,
    obstacle_speed_limit,
    speed_command,
)

STRAIGHT = [(float(x), 0.0) for x in range(1, 60, 3)]


def test_prune_drops_points_behind():
    wps = [(-3.0, 0.0), (-0.5, 0.0), (2.0, 0.0), (5.0, 0.0)]
    prune_passed(wps, 0.0, 0.0, 0.0)
    assert wps == [(2.0, 0.0), (5.0, 0.0)]


def test_lookahead_and_fallback():
    assert find_lookahead(STRAIGHT, 0.0, 0.0, 6.0) == (7.0, 0.0)
    assert find_lookahead([(1.0, 0.0), (2.0, 0.0)], 0.0, 0.0, 6.0) == (2.0, 0.0)
    assert find_lookahead([], 0.0, 0.0, 6.0) is None


def test_steer_sign_follows_carla_convention():
    # Target to the LEFT (+y in ROS) → negative (left) CARLA steer
    assert pure_pursuit_steer(0, 0, 0, (6.0, 2.0), 2.875, 1.0, 0.6) < 0
    # Target to the RIGHT → positive steer
    assert pure_pursuit_steer(0, 0, 0, (6.0, -2.0), 2.875, 1.0, 0.6) > 0
    # Straight ahead → zero
    assert pure_pursuit_steer(0, 0, 0, (6.0, 0.0), 2.875, 1.0, 0.6) == 0.0


def test_steer_respects_heading_and_clamp():
    # Facing +y, target further along +y → straight
    assert abs(pure_pursuit_steer(0, 0, math.pi / 2, (0.0, 6.0), 2.875, 1.0, 0.6)) < 1e-9
    # Hard left target is clamped
    assert pure_pursuit_steer(0, 0, 0, (0.5, 3.0), 2.875, 5.0, 0.6) == -0.6


def test_obstacle_in_corridor_detected_with_arc_length():
    hit = obstacle_along_path(0, 0, 0, STRAIGHT, [(20.0, 0.5)], 2.0)
    assert hit is not None and abs(hit[0] - 20.0) < 1e-6 and hit[1] == 0.0


def test_obstacle_outside_corridor_or_behind_ignored():
    assert obstacle_along_path(0, 0, 0, STRAIGHT, [(20.0, 3.5)], 2.0) is None   # adjacent lane
    assert obstacle_along_path(0, 0, 0, STRAIGHT, [(-5.0, 0.0)], 2.0) is None   # behind
    assert obstacle_along_path(0, 0, 0, STRAIGHT, [], 2.0) is None


def test_obstacle_on_curved_path_uses_path_not_heading():
    # Path turns left 90°; obstacle sits on the turned section, off the
    # straight-ahead line.
    path = [(float(x), 0.0) for x in range(1, 11)] + [(10.0, float(y)) for y in range(1, 20)]
    hit = obstacle_along_path(0, 0, 0, path, [(10.0, 10.0)], 2.0)
    assert hit is not None and abs(hit[0] - 20.0) < 1e-6


def test_speed_limit_profile():
    assert obstacle_speed_limit(None, 3.0, 8.0, 2.0) == 3.0
    assert obstacle_speed_limit(5.0, 3.0, 8.0, 2.0) == 0.0
    assert obstacle_speed_limit(100.0, 3.0, 8.0, 2.0) == 3.0
    v = obstacle_speed_limit(9.0, 3.0, 8.0, 2.0)
    assert 0.0 < v < 3.0


def test_speed_command():
    thr, brk = speed_command(0.0, 3.0, 0.5, 0.5)
    assert thr > 0 and brk == 0
    assert speed_command(3.2, 3.0, 0.5, 0.5) == (0.0, 0.0)       # small overspeed: coast
    thr, brk = speed_command(5.0, 1.0, 0.5, 0.5)
    assert thr == 0 and brk > 0
    assert speed_command(0.0, 0.0, 0.5, 0.5) == (0.0, 1.0)       # hold at stop


def test_governor_filters_single_frame_noise():
    from av_control.pure_pursuit import ObstacleSpeedGovernor
    g = ObstacleSpeedGovernor(stop_distance=8.0, comfort_decel=2.0, window=5)
    for _ in range(5):
        g.update(None, 3.0)
    # a single spurious detection does not slow the vehicle
    assert g.update(9.0, 3.0) == 3.0


def test_governor_holds_stop_until_resume_margin():
    from av_control.pure_pursuit import ObstacleSpeedGovernor
    g = ObstacleSpeedGovernor(stop_distance=8.0, comfort_decel=2.0, resume_margin=2.0, window=1)
    assert g.update(7.5, 3.0) == 0.0
    assert g.update(8.4, 3.0) == 0.0      # noise above the stop line: keep holding
    assert g.update(9.9, 3.0) == 0.0
    assert g.update(10.5, 3.0) > 0.0      # obstacle moved away: release
    assert g.update(None, 3.0) == 3.0


def test_governor_does_not_release_on_single_clear_frame():
    from av_control.pure_pursuit import ObstacleSpeedGovernor
    g = ObstacleSpeedGovernor(stop_distance=8.0, comfort_decel=2.0, resume_margin=2.0, window=5)
    for _ in range(5):
        g.update(6.0, 3.0)
    assert g.holding
    # obstacle flickers out of the corridor for a frame or two
    for d in (None, 6.0, None, None, 6.0):
        assert g.update(d, 3.0) == 0.0
    for _ in range(5):
        v = g.update(None, 3.0)
    assert v == 3.0 and not g.holding


# ── Boxes, prediction, lead vehicles ──────────────────────────────────────────

def car(x, y, yaw=0.0, vx=0.0, vy=0.0):
    return (x, y, yaw, 2.2, 0.9, vx, vy)


def test_box_width_counts_toward_corridor():
    # centre 2.0 m off the path: a point is outside a 1.35 m corridor,
    # but a car's edge (0.9 m half-width) is 1.1 m off → inside
    assert obstacle_along_path(0, 0, 0, STRAIGHT, [(20.0, 2.0)], 1.35) is None
    assert obstacle_along_path(0, 0, 0, STRAIGHT, [car(20.0, 2.0)], 1.35) is not None


def test_gap_is_measured_to_box_minus_ego_half_length():
    gap, _ = obstacle_along_path(0, 0, 0, STRAIGHT, [car(30.0, 0.0)], 1.35, ego_half_length=2.35)
    assert gap == pytest.approx(30.0 - 2.2 - 2.35)


def test_crossing_pedestrian_is_predicted_into_the_corridor():
    walker_static = (15.0, 4.0, 0.0, 0.3, 0.3, 0.0, 0.0)
    walker_crossing = (15.0, 4.0, 0.0, 0.3, 0.3, 0.0, -1.5)   # walking toward the path
    kw = dict(corridor_half_width=1.35, ego_speed=3.0)
    assert obstacle_along_path(0, 0, 0, STRAIGHT, [walker_static], **kw) is None
    assert obstacle_along_path(0, 0, 0, STRAIGHT, [walker_crossing], **kw) is not None


def test_lead_vehicle_velocity_and_credit():
    gap, v_along = obstacle_along_path(0, 0, 0, STRAIGHT, [car(20.0, 0.0, vx=2.0)], 1.35)
    assert v_along == pytest.approx(2.0)
    assert effective_gap(gap, v_along, 2.0) == pytest.approx(gap + 1.0)     # 2²/(2·2)
    assert effective_gap(gap, -2.0, 2.0) == pytest.approx(gap)              # oncoming: no credit
    assert effective_gap(None, 0.0, 2.0) is None


def test_path_length_ahead():
    assert path_length_ahead(0, 0, [(3.0, 4.0), (3.0, 10.0)]) == pytest.approx(11.0)
    assert path_length_ahead(0, 0, []) == 0.0


def circle_path(radius, n=40, step=1.0):
    return [(radius * math.sin(i * step / radius), radius * (1 - math.cos(i * step / radius)))
            for i in range(1, n)]


def test_path_curvature_ahead():
    assert path_curvature_ahead(0, 0, STRAIGHT, 30.0) < 1e-6
    assert path_curvature_ahead(0, 0, circle_path(10.0), 20.0) == pytest.approx(0.1, rel=0.05)


def test_adaptive_lookahead():
    kw = dict(time_s=2.0, min_m=3.0, max_m=12.0, curvature_gain=0.35)
    assert adaptive_lookahead(3.0, 0, 0, STRAIGHT, **kw) == pytest.approx(6.0)     # 2 s at 3 m/s
    assert adaptive_lookahead(0.5, 0, 0, STRAIGHT, **kw) == pytest.approx(3.0)     # min clamp
    assert adaptive_lookahead(3.0, 0, 0, circle_path(10.0), **kw) == pytest.approx(3.5, rel=0.05)
    # cap not binding on a gentle curve
    assert adaptive_lookahead(3.0, 0, 0, circle_path(30.0), **kw) == pytest.approx(6.0)


def test_slow_velocity_noise_does_not_sweep_parked_car_into_corridor():
    # parked car in the adjacent lane with 0.3 m/s of velocity noise toward us
    parked = (20.0, 3.5, 0.0, 2.2, 0.9, 0.0, -0.3)
    assert obstacle_along_path(0, 0, 0, STRAIGHT, [parked], 1.35, ego_speed=3.0) is None


# ── Lead-car following ───────────────────────────────────────────────────────

def test_governor_follows_moving_lead_at_time_gap_without_latching():
    from av_control.pure_pursuit import ObstacleSpeedGovernor
    g = ObstacleSpeedGovernor(stop_distance=3.5, comfort_decel=2.0, window=1)
    d_des = 3.5 + 1.0 * 1.5
    assert g.update(d_des, 3.0, lead_speed=1.5) == pytest.approx(1.5)   # equilibrium
    assert g.update(d_des + 2.0, 3.0, lead_speed=1.5) > 1.5             # catch up
    assert g.update(d_des - 1.0, 3.0, lead_speed=1.5) < 1.5             # fall back
    # noise dips below the stop line: slows, but does not latch a stop
    assert g.update(3.3, 3.0, lead_speed=1.5) >= 0.0 and not g.holding
    assert g.update(d_des, 3.0, lead_speed=1.5) == pytest.approx(1.5)


def test_governor_switches_to_stop_and_hold_when_lead_parks():
    from av_control.pure_pursuit import ObstacleSpeedGovernor
    g = ObstacleSpeedGovernor(stop_distance=3.5, comfort_decel=2.0, window=1)
    g.update(6.0, 3.0, lead_speed=1.5)
    assert g.update(3.4, 3.0, lead_speed=0.0) == 0.0 and g.holding
    assert g.update(4.5, 3.0, lead_speed=0.0) == 0.0                    # held until resume line


def test_heading_corridor_catches_lead_the_plan_has_already_left():
    # mid lane change: the plan is in the left lane, the car still half behind the lead
    plan = [(float(x), 3.5) for x in range(2, 60, 3)]
    lead = car(6.5, 0.0, vx=1.0)
    kw = dict(corridor_half_width=1.35, ego_half_length=2.4)
    assert obstacle_along_path(0, 1.5, 0, plan, [lead], **kw) is None          # plan only
    gap, _ = obstacle_along_path(0, 1.5, 0, plan, [lead], heading_check_m=4.5, **kw)
    assert gap == pytest.approx(6.5 - 2.2 - 2.4)
    # once the car is in the target lane the lead no longer blocks
    assert obstacle_along_path(0, 3.3, 0, plan, [lead], heading_check_m=4.5, **kw) is None


def _arc(radius, n=40, step=1.0, y0=0.0):
    """Left arc from (0, y0) heading +x."""
    out = []
    for i in range(1, n):
        a = i * step / radius
        out.append((radius * math.sin(a), y0 + radius * (1 - math.cos(a))))
    return out


def test_sustained_curvature_curve_vs_lane_change():
    from av_control.pure_pursuit import sustained_curvature
    assert sustained_curvature([(0.0, 0.0)] + _arc(26.5)) == pytest.approx(1 / 26.5, rel=0.1)
    # lane change S on a straight: turns left then right
    s_curve = [(float(x), 3.5 * (3 * (x / 20) ** 2 - 2 * (x / 20) ** 3)) for x in range(1, 21)]
    s_curve += [(float(x), 3.5) for x in range(21, 40)]
    assert sustained_curvature([(0.0, 0.0)] + s_curve) == 0.0


def test_heading_corridor_follows_the_curve():
    # ego in the inner lane of a left curve; a car level-ish in the outer lane
    # (3.5 m to the right along the curve), ~9 m ahead: not in our way
    R = 26.5
    plan = _arc(R)
    a = 9.0 / R
    rx, ry = (R + 3.5) * math.sin(a), R - (R + 3.5) * math.cos(a)
    other = (rx, ry, a, 2.2, 0.9, 1.0 * math.cos(a), 1.0 * math.sin(a))
    kw = dict(corridor_half_width=1.35, ego_half_length=2.4, heading_check_m=4.5)
    assert obstacle_along_path(0, 0, 0, plan, [other], **kw) is None


def test_governor_keeps_following_a_slow_noisy_lead():
    # 0.6 m/s lead, perception noise 0.2 m/s / 0.4 m: without hysteresis the
    # median speed dips under 0.5 m/s, stop-and-hold latches and the car
    # stop-starts in 92 of 100 seeds; with it, 4 (a lead this close to the
    # parked threshold is a known limitation, so this is a rate, not "never").
    import random
    from av_control.pure_pursuit import ObstacleSpeedGovernor
    stopped = 0
    for seed in range(100):
        rng = random.Random(seed)
        g = ObstacleSpeedGovernor(stop_distance=3.5, comfort_decel=1.5)
        gap, v_ego = 5.0, 0.6
        for _ in range(600):          # 60 s at 10 Hz
            v_cmd = g.update(gap + rng.gauss(0.0, 0.4), 3.0,
                             lead_speed=0.6 + rng.gauss(0.0, 0.2))
            v_ego += max(-0.15, min(0.15, v_cmd - v_ego))
            gap += (0.6 - v_ego) * 0.1
            if v_ego <= 0.05:
                stopped += 1
                break
    assert stopped <= 10


def test_governor_parity_reference():
    # Deterministic noise; av_control_cpp's test_safety.cpp checks the C++
    # port against these exact values.
    from av_control.pure_pursuit import ObstacleSpeedGovernor
    g = ObstacleSpeedGovernor(stop_distance=3.5, comfort_decel=1.5)
    gap, v = 5.0, 0.6
    for i in range(600):
        c = g.update(gap + 0.4 * math.sin(1.7 * i), 3.0,
                     lead_speed=0.6 + 0.2 * math.sin(2.3 * i + 1.0))
        v += max(-0.15, min(0.15, c - v))
        gap += (0.6 - v) * 0.1
    assert c == pytest.approx(0.6192893695862955, abs=1e-12)
    assert gap == pytest.approx(4.108379436166265, abs=1e-12)


def test_slow_lead_reports_its_raw_speed():
    lead = car(10.0, 0.0, vx=0.4)       # below the 0.5 m/s "parked" threshold
    gap, v_along = obstacle_along_path(0, 0, 0, [(float(x), 0.0) for x in range(1, 30)],
                                       [lead], corridor_half_width=1.35, ego_half_length=2.4)
    assert v_along == pytest.approx(0.4)


def test_oncoming_car_on_a_curve_is_predicted_along_its_lane():
    # ego in the outer lane of a left curve (R 30 m); an oncoming car in the
    # inner lane (3.5 m left) ~25 m ahead: its tangent crosses our lane, its lane doesn't
    R = 30.0
    plan = _arc(R, n=60)
    a = 25.0 / R
    ox, oy = (R - 3.5) * math.sin(a), R - (R - 3.5) * math.cos(a)
    yaw = a + math.pi
    v = 4.0
    oncoming = (ox, oy, yaw, 2.2, 0.9, v * math.cos(yaw), v * math.sin(yaw))
    kw = dict(corridor_half_width=1.35, ego_half_length=2.4, ego_speed=2.0,
              heading_check_m=4.5)
    assert obstacle_along_path(0, 0, 0, plan, [oncoming], **kw) is None
    # ...but a car cutting in across the path is still caught
    cut_in = (10.0, 3.0, -0.5, 2.2, 0.9, 1.0, -1.0)
    plan_straight = [(float(x), 0.0) for x in range(1, 40)]
    assert obstacle_along_path(0, 0, 0, plan_straight, [cut_in], **kw) is not None
