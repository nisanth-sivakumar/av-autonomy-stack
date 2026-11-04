import math
from types import SimpleNamespace

import numpy as np
import pytest

from av_planning.behavior import ONCOMING, SAME
from av_planning.lanes import lane_direction, neighbour_lane, wp_pose


class StubWP:
    """CARLA-style waypoint on a road given by a centre-line function (ROS frame)."""

    def __init__(self, road, s, lane_id, lanes):
        self.road, self.s, self.lane_id, self.lanes = road, s, lane_id, lanes
        self.lane_type = "Driving"
        self.lane_width = 3.5
        x, y, yaw = road(s, lanes[lane_id])
        if lane_id > 0:                           # oncoming lanes point the other way
            yaw += math.pi
        self.transform = SimpleNamespace(
            location=SimpleNamespace(x=x, y=-y, z=0.0),
            rotation=SimpleNamespace(yaw=-math.degrees(yaw)))

    def _neighbour(self, delta):
        nid = self.lane_id + delta
        if nid == 0:                              # skip the reference line
            nid += delta
        return StubWP(self.road, self.s, nid, self.lanes) if nid in self.lanes else None

    # CARLA: left/right relative to the lane's own direction (lane -1's left is +1)
    def get_left_lane(self):
        return self._neighbour(+1 if self.lane_id < 0 else -1)

    def get_right_lane(self):
        return self._neighbour(-1 if self.lane_id < 0 else +1)


def straight(s, lateral):
    return s, lateral, 0.0


def curve(s, lateral, r=30.0):
    a = s / r
    return (r * math.sin(a) - lateral * math.sin(a), r * (1 - math.cos(a)) + lateral * math.cos(a), a)


def route(road, lane_id, lanes, n=20):
    return [StubWP(road, 3.0 * i, lane_id, lanes) for i in range(n)]


DRIVING = lambda wp: wp.lane_type == "Driving"  # noqa: E731


def test_lane_direction_from_lane_id_sign():
    assert lane_direction(-2, -1) == SAME
    assert lane_direction(-1, 1) == ONCOMING
    assert lane_direction(2, 1) == SAME


@pytest.mark.parametrize("road", [straight, curve])
def test_same_direction_left_neighbour(road):
    # two same-direction lanes: ego in lane -2 (lateral 0), lane -1 to its left
    lanes = {-2: 0.0, -1: 3.5}
    r = route(road, -2, lanes)
    left, wps = neighbour_lane(r, "left", DRIVING)
    assert left.direction == SAME and len(wps) == len(r)
    for (x, y, yaw), ego in zip(left.centre, r):
        ex, ey, eyaw = wp_pose(ego)
        lat = -(x - ex) * math.sin(eyaw) + (y - ey) * math.cos(eyaw)
        assert lat == pytest.approx(3.5) and yaw == pytest.approx(eyaw)
    assert neighbour_lane(r, "right", DRIVING)[0] is None


@pytest.mark.parametrize("road", [straight, curve])
def test_oncoming_left_neighbour(road):
    lanes = {-1: 0.0, 1: 3.5}
    left, _ = neighbour_lane(route(road, -1, lanes), "left", DRIVING)
    assert left.direction == ONCOMING


def test_neighbour_truncated_where_lane_ends_and_min_length():
    lanes = {-2: 0.0, -1: 3.5}
    r = route(straight, -2, lanes)
    for wp in r[6:]:                               # the left lane ends after 6 waypoints
        wp.lanes = {-2: 0.0}
    left, wps = neighbour_lane(r, "left", DRIVING)
    assert left is not None and len(wps) == 6
    assert neighbour_lane(r, "left", DRIVING, min_length=7)[0] is None


def test_non_driving_neighbour_is_ignored():
    lanes = {-2: 0.0, -1: 3.5}
    r = route(straight, -2, lanes)
    orig = StubWP.get_left_lane

    def shoulder(self):
        n = orig(self)
        if n is not None:
            n.lane_type = "Shoulder"
        return n
    StubWP.get_left_lane = shoulder
    try:
        assert neighbour_lane(r, "left", DRIVING)[0] is None
    finally:
        StubWP.get_left_lane = orig
