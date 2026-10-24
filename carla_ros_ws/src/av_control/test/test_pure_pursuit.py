import math

from av_control.pure_pursuit import (
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
    d = obstacle_along_path(0, 0, 0, STRAIGHT, [(20.0, 0.5)], 2.0)
    assert d is not None and abs(d - 20.0) < 1e-6


def test_obstacle_outside_corridor_or_behind_ignored():
    assert obstacle_along_path(0, 0, 0, STRAIGHT, [(20.0, 3.5)], 2.0) is None   # adjacent lane
    assert obstacle_along_path(0, 0, 0, STRAIGHT, [(-5.0, 0.0)], 2.0) is None   # behind
    assert obstacle_along_path(0, 0, 0, STRAIGHT, [], 2.0) is None


def test_obstacle_on_curved_path_uses_path_not_heading():
    # Path turns left 90°; obstacle sits on the turned section, off the
    # straight-ahead line.
    path = [(float(x), 0.0) for x in range(1, 11)] + [(10.0, float(y)) for y in range(1, 20)]
    d = obstacle_along_path(0, 0, 0, path, [(10.0, 10.0)], 2.0)
    assert d is not None and abs(d - 20.0) < 1e-6


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
