"""
Randomized scenario sweep for run_scenarios.py (--sweep N --seed S).

Each variant is drawn from one of four families, round robin, with its own
RNG seeded by (seed, index), so `--sweep N --seed S` always produces the same
variants (Python's random is stable across versions for these calls):

    follow    single lane, lead car at a random distance and speed (< target)
    parked    single lane, parked car at a random position on the straight or
              curve, from fully blocking to clear of the lane
    overtake  two same-direction lanes, slow lead; sometimes a second car in
              the left lane (random position and speed, maybe speeding up
              when the ego leaves its lane)
    oncoming  two lanes, the other one oncoming with traffic; slow lead

Every variant also randomizes the ego's start offset (±0.4 m) and heading
error (±5°).

Checks are of two kinds:
  * invariants that must hold whatever was drawn (no collision, bumper gap,
    side clearance, never into an oncoming lane);
  * liveness checks only where the expected behaviour is unambiguous (a car
    clear of the lane must be passed without stopping; a free target lane
    must be used to overtake).

The parameters, not the RNG state, are what a failure records, so
`--replay failures.json` re-runs exactly the same scenario.  The closed loop
is not bit-for-bit deterministic (ROS scheduling, perception noise, MPPI
sampling), so a replay reproduces the situation, not necessarily the outcome.
"""

import random

FAMILIES = ("follow", "parked", "overtake", "oncoming")
TARGET_SPEED = 3.0


def _r(rng, lo, hi, nd=2):
    return round(rng.uniform(lo, hi), nd)


def sample(seed: int, index: int) -> dict:
    """Parameters of variant `index` of sweep `seed` (JSON-serialisable)."""
    rng = random.Random(f"{seed}-{index}")
    family = FAMILIES[index % len(FAMILIES)]
    p = {"family": family, "seed": seed, "index": index,
         "start_lateral": _r(rng, -0.4, 0.4), "start_yaw_deg": _r(rng, -5.0, 5.0, 1)}
    if family == "follow":
        p.update(lead_s=_r(rng, 18, 40, 1), lead_speed=_r(rng, 0.8, 2.4))
    elif family == "parked":
        p.update(obs_s=_r(rng, 28, 65, 1), obs_lateral=_r(rng, -3.0, 3.0))
    elif family == "overtake":
        p.update(lead_s=_r(rng, 20, 35, 1), lead_speed=_r(rng, 0.5, 1.6))
        if rng.random() < 0.5:
            p.update(other_s=_r(rng, -15, 45, 1), other_speed=_r(rng, 0.5, 4.0))
            if rng.random() < 0.3:
                p["other_trigger_speed"] = _r(rng, 4.0, 7.0)
    else:  # oncoming
        p.update(lead_s=_r(rng, 20, 35, 1), lead_speed=_r(rng, 0.5, 1.6),
                 other_s=_r(rng, 60, 150, 1), other_speed=_r(rng, 2.0, 5.0))
    return p


def name(p: dict) -> str:
    return f"{p['family']}_s{p['seed']}_{p['index']:03d}"


def _expect(checks):
    return [label for label, ok in checks if not ok]


def _v(x, default):
    return default if x is None else x


def build(p: dict) -> dict:
    """run_scenarios spec (scenario JSON, start pose, duration, check) for `p`."""
    fam = p["family"]
    start = {"s": 5.0, "lateral": p["start_lateral"], "yaw_deg": p["start_yaw_deg"]}

    if fam == "follow":
        scen = {"obstacles": [{"s": p["lead_s"], "lateral": 0.0, "speed": p["lead_speed"]}]}

        def check(r, planner):
            return _expect([
                ("collisions", r["collisions"] == 0),
                ("bumper gap ≥ 2 m", _v(r["min_lead_gap_m"], 99) >= 2.0),
                ("still moving", r["final_speed_mps"] > 0.5),
                ("stayed in lane (|final lateral| < 0.6 m)", abs(r["final_lateral_m"]) < 0.6),
            ])
        return dict(scenario=scen, start=start, duration=40.0, check=check)

    if fam == "parked":
        scen = {"obstacles": [{"s": p["obs_s"], "lateral": p["obs_lateral"]}]}
        # box edge ≥ 1.75 m from the centre line (lattice min clearance): the
        # car is clear of the controller corridor, so nobody may stop for it
        clear = abs(p["obs_lateral"]) - 0.85 >= 1.75

        def check(r, planner):
            passed = r["final_s_m"] > p["obs_s"] + 5.0
            stopped = r["final_speed_mps"] < 0.1
            return _expect([
                ("collisions", r["collisions"] == 0),
                ("passed or stopped short (≥ 1.5 m)",
                 passed or (stopped and _v(r["min_lead_gap_m"], 99) >= 1.5)),
                ("side clearance ≥ 0.2 m when passing",
                 not passed or _v(r["min_side_clearance_m"], 9) >= 0.2),
            ] + ([("passed a car clear of the lane", passed)] if clear else []))
        return dict(scenario=scen, start=start, duration=40.0, check=check)

    if fam == "overtake":
        obs = [{"s": p["lead_s"], "lateral": 0.0, "speed": p["lead_speed"]}]
        other = "other_s" in p
        if other:
            o = {"s": p["other_s"], "lateral": 3.5, "speed": p["other_speed"]}
            if "other_trigger_speed" in p:
                o["trigger_speed"] = p["other_trigger_speed"]
            obs.append(o)
        scen = {"lanes": "same", "obstacles": obs}

        def check(r, planner):
            lead_final = (r.get("obstacle_final_s_m") or [0.0])[0]
            checks = [
                ("collisions", r["collisions"] == 0),
                ("bumper gap ≥ 1.5 m", _v(r["min_lead_gap_m"], 99) >= 1.5),
                ("side clearance ≥ 0.3 m", _v(r["min_side_clearance_m"], 9) >= 0.3),
            ]
            if planner == "road_graph":
                checks.append(("no lane change", r["lane_changes"] == 0))
            elif not other:
                checks += [("changed lane", r["lane_changes"] >= 1),
                           ("passed the lead", r["final_s_m"] > lead_final + 5.0)]
            return _expect(checks)
        return dict(scenario=scen, start=start, duration=45.0, check=check)

    # oncoming
    scen = {"lanes": "oncoming", "obstacles": [
        {"s": p["lead_s"], "lateral": 0.0, "speed": p["lead_speed"]},
        {"s": p["other_s"], "lateral": 3.5, "speed": p["other_speed"], "oncoming": True}]}

    def check(r, planner):
        return _expect([
            ("collisions", r["collisions"] == 0),
            ("no lane change", r["lane_changes"] == 0),
            ("stayed in lane (max |lateral| < 1.0 m)", r["max_lateral_m"] < 1.0),
            # a lead just above the 0.5 m/s "parked" threshold can read as
            # parked for a moment before following latches (seen: 1.0 s)
            ("kept following (stopped < 2 s)", r["stopped_time_s"] < 2.0),
        ])
    return dict(scenario=scen, start=start, duration=40.0, check=check)
