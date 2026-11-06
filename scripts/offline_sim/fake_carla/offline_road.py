"""
Synthetic road + scenario shared by the fake `carla` module and sim_node.

Geometry is defined in the ROS map frame (x forward, y left):
    straight 30 m along +x  →  90° left arc, radius R  →  straight 200 m along +y
Lanes are 3.5 m wide.  The ego drives along increasing arc length s in the
lane at lateral 0.  "lanes" in the scenario selects the layout:
    "single"   : one lane (lane_id -1)
    "same"     : ego lane -2, plus a same-direction lane -1 3.5 m to its left
    "oncoming" : ego lane -1, plus an oncoming lane +1 3.5 m to its left

The scenario is read from the JSON file named by OFFLINE_SIM_SCENARIO:
    {
      "radius": 30.0,                                   # optional, curve radius
      "obstacles": [
        {"s": 45.0, "lateral": 0.0,                     # + = left of centre
         "speed": 0.0,                                  # optional, along the lane (m/s)
         "oncoming": false,                             # optional, drives towards -s
         "trigger_speed": 6.0,                          # optional, speed after the trigger
         "half_length": 2.0, "half_width": 0.85}        # optional (Audi A2-ish)
      ]
    }

Moving obstacles share a *simulated* time base across processes
(perception, planner, sim): sim_node writes the drive time (simulated
seconds since the ego started driving) to the file named by
OFFLINE_SIM_CLOCK every step, and obstacle s = s0 + speed·t.  Being on
simulated time, the whole loop can run faster than real time
(sim_node --rate).  It writes the drive time at which the ego first leaves
its lane (|lateral| > 0.5 m, i.e. a lane change starts) to
OFFLINE_SIM_TRIGGER; obstacles with "trigger_speed" switch to that speed
then (used to close a gap mid-change).
"""

import json
import math
import os

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
LAYOUTS = {
    "single": {-1: 0.0},
    "same": {-2: 0.0, -1: LANE_WIDTH},
    "oncoming": {-1: 0.0, 1: LANE_WIDTH},
}
LANES = LAYOUTS[scenario().get("lanes", "single")]     # lane_id → lateral offset
EGO_LANE_ID = min(LANES, key=lambda k: abs(LANES[k]))   # the lane at lateral 0


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
            "oncoming": bool(o.get("oncoming", False)),
            "trigger_speed": o.get("trigger_speed"),
        })
    return out


def _read_time(env: str):
    path = os.environ.get(env)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return None          # mid-write: caller falls back


_last_clock = 0.0


def elapsed() -> float:
    """Simulated seconds since the ego started driving (0 before that)."""
    global _last_clock
    t = _read_time("OFFLINE_SIM_CLOCK")
    if t is not None:
        _last_clock = max(0.0, t)
    return _last_clock


def trigger_elapsed() -> float:
    """Simulated seconds since the trigger (ego left its lane), or None."""
    t_trig = _read_time("OFFLINE_SIM_TRIGGER")
    return None if t_trig is None else max(0.0, elapsed() - t_trig)


def obstacle_state(spec: dict, t: float, t_trig: float = None) -> dict:
    """Obstacle pose and velocity at time t (ROS frame); t_trig = time since trigger."""
    v = spec["speed"]
    travelled = v * t
    if spec.get("trigger_speed") is not None and t_trig is not None:
        t_trig = min(t_trig, t)
        travelled = v * (t - t_trig) + float(spec["trigger_speed"]) * t_trig
        v = float(spec["trigger_speed"])
    sign = -1.0 if spec.get("oncoming") else 1.0
    s = max(0.0, min(LENGTH, spec["s"] + sign * travelled))
    x, y = point_at(s, spec["lateral"])
    _, _, yaw = pose_at(s)
    if spec.get("oncoming"):
        yaw += math.pi
    moving = v > 0 and 0.0 < s < LENGTH
    v = v if moving else 0.0
    return {"s": s, "lateral": spec["lateral"], "x": x, "y": y, "yaw": yaw,
            "vx": v * math.cos(yaw), "vy": v * math.sin(yaw)}


def obstacle_now(spec: dict) -> dict:
    return obstacle_state(spec, elapsed(), trigger_elapsed())


def load_obstacles() -> list[tuple[float, float]]:
    """Obstacle centres (ROS frame) right now."""
    return [(st["x"], st["y"]) for st in (obstacle_now(o) for o in obstacle_specs())]
