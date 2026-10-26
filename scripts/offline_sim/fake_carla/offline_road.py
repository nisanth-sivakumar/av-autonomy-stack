"""
Synthetic road + scenario shared by the fake `carla` module and sim_node.

Geometry is defined in the ROS map frame (x forward, y left):
    straight 30 m along +x  →  90° left arc, radius 30 m  →  straight 200 m along +y
One driving lane, 3.5 m wide, driving direction = increasing arc length s.

The scenario (obstacles) is read from the JSON file named by the
OFFLINE_SIM_SCENARIO environment variable:
    {"obstacles": [{"s": 45.0, "lateral": 0.0}, ...]}
`lateral` is + left of the lane centre.
"""

import json
import math
import os

STRAIGHT_1 = 30.0
RADIUS = 30.0
ARC = RADIUS * math.pi / 2
STRAIGHT_2 = 200.0
LENGTH = STRAIGHT_1 + ARC + STRAIGHT_2
LANE_WIDTH = 3.5


def pose_at(s: float) -> tuple[float, float, float]:
    """(x, y, yaw) of the lane centre at arc length s, ROS frame."""
    s = max(0.0, min(LENGTH, s))
    if s <= STRAIGHT_1:
        return s, 0.0, 0.0
    if s <= STRAIGHT_1 + ARC:
        a = (s - STRAIGHT_1) / RADIUS
        return STRAIGHT_1 + RADIUS * math.sin(a), RADIUS * (1 - math.cos(a)), a
    d = s - STRAIGHT_1 - ARC
    return STRAIGHT_1 + RADIUS, RADIUS + d, math.pi / 2


def point_at(s: float, lateral: float) -> tuple[float, float]:
    x, y, yaw = pose_at(s)
    return x - lateral * math.sin(yaw), y + lateral * math.cos(yaw)


def project(x: float, y: float) -> tuple[float, float]:
    """Closest (s, lateral) on the centre-line for a ROS-frame point."""
    best = (float("inf"), 0.0, 0.0)
    # coarse search then refine
    s = 0.0
    while s <= LENGTH:
        cx, cy, _ = pose_at(s)
        d = math.hypot(x - cx, y - cy)
        if d < best[0]:
            best = (d, s, 0.0)
        s += 1.0
    lo, hi = max(0.0, best[1] - 1.0), min(LENGTH, best[1] + 1.0)
    for _ in range(40):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        d1 = math.hypot(x - pose_at(m1)[0], y - pose_at(m1)[1])
        d2 = math.hypot(x - pose_at(m2)[0], y - pose_at(m2)[1])
        if d1 < d2:
            hi = m2
        else:
            lo = m1
    s = (lo + hi) / 2
    cx, cy, yaw = pose_at(s)
    lateral = (x - cx) * -math.sin(yaw) + (y - cy) * math.cos(yaw)
    return s, lateral


def load_obstacles() -> list[tuple[float, float]]:
    """Obstacle centres (ROS frame) from the scenario file."""
    path = os.environ.get("OFFLINE_SIM_SCENARIO")
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        spec = json.load(f)
    return [point_at(o["s"], o.get("lateral", 0.0)) for o in spec.get("obstacles", [])]
