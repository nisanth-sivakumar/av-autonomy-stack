"""
Synthetic road + scenario shared by the fake `carla` module and sim_node.

Geometry is defined in the ROS map frame (x forward, y left):
    straight 30 m along +x  →  90° left arc, radius R  →  straight 200 m along +y
One driving lane, 3.5 m wide, driving direction = increasing arc length s.

The scenario is read from the JSON file named by OFFLINE_SIM_SCENARIO:
    {
      "radius": 30.0,                                   # optional, curve radius
      "obstacles": [
        {"s": 45.0, "lateral": 0.0,                     # + = left of centre
         "speed": 0.0,                                  # optional, along the lane (m/s)
         "half_length": 2.0, "half_width": 0.85}        # optional (Audi A2-ish)
      ]
    }

Moving obstacles share a time base across processes (perception, planner,
sim): sim_node writes the wall-clock time at which the ego starts driving
to the file named by OFFLINE_SIM_EPOCH, and obstacle s = s0 + speed·t.
"""

import json
import math
import os
import time

_SPEC_CACHE = None


def scenario() -> dict:
    global _SPEC_CACHE
    if _SPEC_CACHE is None:
        path = os.environ.get("OFFLINE_SIM_SCENARIO")
        _SPEC_CACHE = {}
        if path and os.path.exists(path):
            with open(path) as f:
                _SPEC_CACHE = json.load(f)
    return _SPEC_CACHE


STRAIGHT_1 = 30.0
RADIUS = float(scenario().get("radius", 30.0))
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
    best = (float("inf"), 0.0)
    s = 0.0
    while s <= LENGTH:
        cx, cy, _ = pose_at(s)
        d = math.hypot(x - cx, y - cy)
        if d < best[0]:
            best = (d, s)
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


# ── Obstacles ─────────────────────────────────────────────────────────────

def obstacle_specs() -> list[dict]:
    out = []
    for o in scenario().get("obstacles", []):
        out.append({
            "s": float(o["s"]),
            "lateral": float(o.get("lateral", 0.0)),
            "speed": float(o.get("speed", 0.0)),
            "half_length": float(o.get("half_length", 2.0)),
            "half_width": float(o.get("half_width", 0.85)),
        })
    return out


def elapsed() -> float:
    """Seconds since the ego started driving (0 before that)."""
    path = os.environ.get("OFFLINE_SIM_EPOCH")
    if not path or not os.path.exists(path):
        return 0.0
    try:
        with open(path) as f:
            return max(0.0, time.time() - float(f.read().strip()))
    except (OSError, ValueError):
        return 0.0


def obstacle_state(spec: dict, t: float) -> dict:
    """Obstacle pose and velocity at time t (ROS frame)."""
    s = min(LENGTH, spec["s"] + spec["speed"] * t)
    x, y = point_at(s, spec["lateral"])
    _, _, yaw = pose_at(s)
    moving = spec["speed"] > 0 and s < LENGTH
    v = spec["speed"] if moving else 0.0
    return {"s": s, "lateral": spec["lateral"], "x": x, "y": y, "yaw": yaw,
            "vx": v * math.cos(yaw), "vy": v * math.sin(yaw)}


def load_obstacles() -> list[tuple[float, float]]:
    """Obstacle centres (ROS frame) right now."""
    t = elapsed()
    return [(st["x"], st["y"]) for st in (obstacle_state(o, t) for o in obstacle_specs())]
