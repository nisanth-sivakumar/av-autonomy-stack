import math

import numpy as np
import pytest

from av_planning.behavior import (
    ABORT, CHANGING, FOLLOW, LANE_KEEP, ONCOMING, PREPARE_CHANGE, SAME,
    BehaviorConfig, BehaviorPlanner, Lane, lead_in_lane, project, target_lane_clear,
)


def straight_lane(y, direction=SAME):
    xs = np.arange(-50.0, 400.0, 3.0)
    return Lane(np.column_stack([xs, np.full_like(xs, y), np.zeros_like(xs)]), direction)


def car(x, y, vx=0.0):
    return (x, y, 0.0, 2.0, 0.85, vx, 0.0)


class World:
    """Ego driving along +x; obstacles at constant velocity."""

    def __init__(self, cars, left=SAME, ego_speed=1.0, dt=0.1):
        self.cars = [list(c) for c in cars]
        self.lane_y = {"ego": 0.0, "left": 3.5, "right": None}
        self.left_dir = left
        self.x, self.y, self.v, self.dt, self.t = 0.0, 0.0, ego_speed, dt, 0.0
        self.bp = BehaviorPlanner(BehaviorConfig())

    def lanes(self):
        out = {"ego": straight_lane(self.lane_y["ego"]), "left": None, "right": None}
        if self.lane_y["left"] is not None and self.left_dir is not None:
            out["left"] = straight_lane(self.lane_y["left"], self.left_dir)
        if self.lane_y["right"] is not None:
            out["right"] = straight_lane(self.lane_y["right"])
        return out

    def step(self):
        obs = np.array(self.cars, float).reshape(-1, 7)
        out = self.bp.update(self.t, (self.x, self.y), self.v, self.lanes(), obs)
        ego_y = self.lane_y["ego"]
        if out.target is not None:                 # follow the blend at the ego
            self.y = ego_y + out.progress * (self.lane_y[out.target] - ego_y)
        if out.switch_lane == "left":              # the car is now in the left lane
            self.lane_y = {"ego": self.lane_y["left"], "left": None, "right": ego_y}
            self.y = self.lane_y["ego"]
        self.t += self.dt
        self.x += self.v * self.dt
        for c in self.cars:
            c[0] += c[5] * self.dt
        return out

    def run(self, seconds):
        return [self.step() for _ in range(int(round(seconds / self.dt)))]


def states(outs):
    seq = []
    for o in outs:
        if not seq or seq[-1] != o.state:
            seq.append(o.state)
    return seq


# ── geometry ─────────────────────────────────────────────────────────────────

def test_project_and_lead_detection():
    lane = straight_lane(0.0)
    s, lat, yaw = project(lane.centre, 10.0, 1.0)
    assert s == pytest.approx(60.0) and lat == pytest.approx(1.0) and yaw == pytest.approx(0.0)
    obs = np.array([car(15.0, 0.2, 1.0), car(15.0, 3.5), car(-10.0, 0.0)])
    gap, v = lead_in_lane(lane, (0.0, 0.0), obs, BehaviorConfig())
    assert gap == pytest.approx(15.0 - 2.0 - 2.4) and v == pytest.approx(1.0)


def test_gap_acceptance_cases():
    cfg = BehaviorConfig()
    left = straight_lane(3.5)
    clear = target_lane_clear(left, (0, 0), 1.0, np.zeros((0, 7)), 10.0, cfg)
    assert clear[0]
    ahead = target_lane_clear(left, (0, 0), 1.0, np.array([car(6.0, 3.5, 1.0)]), 10.0, cfg)
    assert not ahead[0] and "ahead" in ahead[1]
    behind = target_lane_clear(left, (0, 0), 1.0, np.array([car(-15.0, 3.5, 5.0)]), 10.0, cfg)
    assert not behind[0] and "behind" in behind[1]
    far_behind_slow = target_lane_clear(left, (0, 0), 1.0, np.array([car(-40.0, 3.5, 1.0)]),
                                        10.0, cfg)
    assert far_behind_slow[0]
    oncoming = target_lane_clear(straight_lane(3.5, ONCOMING), (0, 0), 1.0,
                                 np.zeros((0, 7)), 10.0, cfg)
    assert not oncoming[0]


# ── transitions ──────────────────────────────────────────────────────────────

def test_no_lead_keeps_lane_and_fast_lead_is_just_followed():
    assert states(World([]).run(5)) == [LANE_KEEP]
    w = World([car(15.0, 0.0, 3.0)], ego_speed=3.0)
    assert states(w.run(10)) == [FOLLOW]


def test_slow_lead_triggers_full_lane_change_then_cooldown():
    w = World([car(12.0, 0.0, 0.5)], ego_speed=1.0)
    outs = w.run(40)
    seq = states(outs)
    assert seq[:4] == [FOLLOW, PREPARE_CHANGE, CHANGING, LANE_KEEP]
    switched = [o for o in outs if o.switch_lane]
    assert len(switched) == 1 and switched[0].switch_lane == "left"
    # entered PREPARE only after the lead had been slow for slow_time_s
    t_prepare = next(i for i, o in enumerate(outs) if o.state == PREPARE_CHANGE) * w.dt
    assert t_prepare >= BehaviorConfig().slow_time_s - 1e-6
    # progress rises monotonically during the change
    prog = [o.progress for o in outs if o.state == CHANGING]
    assert prog[0] < 0.1 and all(b >= a - 1e-9 for a, b in zip(prog, prog[1:]))


def test_blocked_target_lane_keeps_following():
    # a car alongside in the target lane moving with the ego
    w = World([car(12.0, 0.0, 0.5), car(1.0, 3.5, 1.0)], ego_speed=1.0)
    seq = states(w.run(20))
    assert CHANGING not in seq and PREPARE_CHANGE in seq


def test_never_changes_into_oncoming_lane():
    w = World([car(12.0, 0.0, 0.5)], left=ONCOMING, ego_speed=1.0)
    seq = states(w.run(20))
    assert seq == [FOLLOW]


def test_closing_gap_mid_change_aborts_back():
    w = World([car(12.0, 0.0, 0.5), car(-60.0, 3.5, 1.0)], ego_speed=1.0)
    outs = []
    for _ in range(400):
        o = w.step()
        outs.append(o)
        if o.state == CHANGING and o.progress > 0.05:
            w.cars[1][5] = 12.0            # the car behind suddenly speeds up
        if o.state == ABORT:
            break
    assert outs[-1].state == ABORT and outs[-1].progress < BehaviorConfig().abort_progress_max
    rest = w.run(15)
    assert rest[-1].progress == pytest.approx(0.0)
    assert not any(o.switch_lane for o in outs + rest)
    assert LANE_KEEP in states(rest) or FOLLOW in states(rest)


def test_past_halfway_the_change_completes():
    w = World([car(12.0, 0.0, 0.5), car(-60.0, 3.5, 1.0)], ego_speed=1.0)
    outs = []
    for _ in range(600):
        o = w.step()
        outs.append(o)
        if o.state == CHANGING and o.progress > 0.6:
            w.cars[1][5] = 12.0
        if o.switch_lane:
            break
    assert outs[-1].switch_lane == "left" and ABORT not in states(outs)


def test_intermittently_slow_lead_does_not_trigger():
    w = World([car(12.0, 0.0, 0.5)], ego_speed=1.0)
    for i in range(100):                      # lead alternates slow/fast every 1.5 s
        w.cars[0][5] = 0.5 if (i // 15) % 2 == 0 else 3.0
        assert w.step().state in (FOLLOW, LANE_KEEP)


def test_weights_profile():
    w = World([car(12.0, 0.0, 0.5)], ego_speed=1.0)
    out = None
    for _ in range(400):
        out = w.step()
        if out.state == CHANGING:
            break
    L = out._length
    assert L == pytest.approx(BehaviorConfig().min_change_length_m)   # slow → short change
    ws = out.weights(np.array([0.0, L / 2, L, 2 * L]))
    assert ws[0] < 0.05 and ws[1] == pytest.approx(0.5, abs=0.05) and ws[2] == ws[3] == 1.0
    assert np.all(np.diff(out.weights(np.linspace(0, L, 50))) >= 0)
    assert np.all(World([]).step().weights(np.linspace(0, 30, 5)) == 0)


def test_noise_does_not_put_the_lead_in_the_target_lane():
    cfg = BehaviorConfig()
    left = straight_lane(3.5)
    # the lead in the ego lane, displaced 0.9 m toward the target lane by noise
    lead = np.array([car(10.0, 0.9, 1.0)])
    assert target_lane_clear(left, (0, 0), 1.0, lead, 10.0, cfg)[0]


def test_single_conflict_frame_does_not_abort():
    w = World([car(12.0, 0.0, 0.5)], ego_speed=1.0)
    outs = []
    for _ in range(400):
        out = w.step()
        outs.append(out)
        if out.state == CHANGING and out.progress > 0.02:
            w.cars.append(list(car(w.x + 6.0, 3.5, 1.0)))     # one-frame ghost in the target lane
            outs.append(w.step())
            w.cars.pop()
            break
    rest = w.run(30)
    assert ABORT not in states(outs + rest)
    assert any(o.switch_lane == "left" for o in rest)
