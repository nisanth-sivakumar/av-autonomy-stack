#!/usr/bin/env python3
"""
Closed-loop offline test of the full stack (perception → planner → controller)
without CARLA.

For each planner × scenario it launches the real ROS nodes with their real
launch files, swaps the CARLA Python API for fake_carla (synthetic road +
parked cars), runs sim_node.py as the vehicle/bridge, and checks the
ground-truth outcome.

Usage (from the repo root, after building carla_ros_ws):
    source carla_ros_ws/install/setup.bash
    python3 scripts/offline_sim/run_scenarios.py                  # everything
    python3 scripts/offline_sim/run_scenarios.py --planners mppi --scenarios partial

Randomized sweep (sweep.py): N variants drawn from seed S, every planner on
each; prints pass rates per family and planner, and saves the failing
variants' parameters to <workdir>/failures.json for --replay:
    python3 scripts/offline_sim/run_scenarios.py --sweep 40 --seed 1 --rate 4 --jobs 3
    python3 scripts/offline_sim/run_scenarios.py --replay <workdir>/failures.json -v
"""

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep  # noqa: E402

# name → spec: scenario JSON for offline_road (radius, obstacles), duration,
# optional goal arc length, and checks(result, planner) → list of failures
SCENARIOS = {
    # Straight → 90° left curve (R 30 m), empty road: lane keeping + perception
    "curve": dict(
        scenario={}, duration=35.0,
        check=lambda r, p: _expect(r, [
            ("collisions", r["collisions"] == 0),
            ("progress > 75 m", r["final_s_m"] > 75),
            ("max |CTE| < 0.6 m", r["max_abs_cte_m"] < 0.6),
            ("lane-offset msgs received", r["lane_offset_msgs"] > 50),
            ("lane-offset mean err < 0.3 m", (r["lane_offset_mean_abs_err_m"] or 9) < 0.3),
        ]),
    ),
    # Junction-like 90° turn with R 10 m: adaptive lookahead keeps CTE down
    "tight": dict(
        scenario={"radius": 10.0}, duration=30.0,
        check=lambda r, p: _expect(r, [
            ("collisions", r["collisions"] == 0),
            ("progress > 60 m", r["final_s_m"] > 60),
            ("max |CTE| < 0.6 m", r["max_abs_cte_m"] < 0.6),
        ]),
    ),
    # Car parked in the middle of the lane on the straight: every planner must stop
    "blocked": dict(
        scenario={"obstacles": [{"s": 28.0, "lateral": 0.0}]}, duration=25.0,
        check=lambda r, p: _expect(r, [
            ("collisions", r["collisions"] == 0),
            ("stopped", r["final_speed_mps"] < 0.1),
            ("bumper gap ≥ 2 m", (r["min_lead_gap_m"] or 0) >= 2.0),
            ("bumper gap ≤ 6 m", (r["min_lead_gap_m"] or 99) <= 6.0),
        ]),
    ),
    # Car parked 1.0 m left of centre on the curve: lattice/MPPI pass it,
    # road_graph (no avoidance) must stop.
    "partial": dict(
        scenario={"obstacles": [{"s": 50.0, "lateral": 1.0}]}, duration=40.0,
        check=lambda r, p: _expect(r, [
            ("collisions", r["collisions"] == 0),
        ] + ([
            ("stopped", r["final_speed_mps"] < 0.1),
        ] if p == "road_graph" else [
            ("passed the obstacle", r["final_s_m"] > 60),
        ])),
    ),
    # Slower lead car (1.5 m/s) in the lane: follow it, don't stop or swerve
    "lead": dict(
        scenario={"obstacles": [{"s": 25.0, "lateral": 0.0, "speed": 1.5}]}, duration=40.0,
        check=lambda r, p: _expect(r, [
            ("collisions", r["collisions"] == 0),
            ("kept a gap ≥ 2 m", (r["min_lead_gap_m"] or 0) >= 2.0),
            ("kept following (progress > 45 m)", r["final_s_m"] > 45),
            ("still moving", r["final_speed_mps"] > 0.8),
            ("stayed in lane (max |CTE| < 0.6 m)", r["max_abs_cte_m"] < 0.6),
        ]),
    ),
    # ── Phase 4.1: behavior layer (two-lane roads) ──
    # Slow lead (1.0 m/s) with a free same-direction lane: lattice/MPPI change
    # lanes and pass; road_graph has no behavior layer and follows.
    "overtake_slow": dict(
        scenario={"lanes": "same", "obstacles": [{"s": 25.0, "lateral": 0.0, "speed": 1.0}]},
        duration=45.0,
        check=lambda r, p: _expect(r, [
            ("collisions", r["collisions"] == 0),
        ] + ([
            ("no lane change", r["lane_changes"] == 0 and r["max_lateral_m"] < 1.0),
            ("kept following", r["final_speed_mps"] > 0.5),
        ] if p == "road_graph" else [
            ("changed lane", r["lane_changes"] >= 1),
            ("bumper gap to the lead ≥ 2 m", (r["min_lead_gap_m"] or 99) >= 2.0),
            ("ends in the left lane", abs(r["final_lateral_m"] - 3.5) < 0.6),
            ("passed the lead (final s > 75 m)", r["final_s_m"] > 75),
            ("no abort", r["aborts"] == 0),
        ])),
    ),
    # Same, but a car in the target lane level with the ego: never change.
    "overtake_blocked": dict(
        scenario={"lanes": "same", "obstacles": [
            {"s": 25.0, "lateral": 0.0, "speed": 1.0},
            {"s": 17.0, "lateral": 3.5, "speed": 1.0}]},
        duration=40.0,
        check=lambda r, p: _expect(r, [
            ("collisions", r["collisions"] == 0),
            ("no lane change", r["lane_changes"] == 0 and r["max_lateral_m"] < 1.0),
            ("kept following", r["final_speed_mps"] > 0.5),
        ]),
    ),
    # The car behind in the target lane speeds up to 6 m/s as the ego starts
    # to leave its lane: abort back, let it pass (a later retry may complete).
    "overtake_abort": dict(
        scenario={"lanes": "same", "obstacles": [
            {"s": 25.0, "lateral": 0.0, "speed": 1.0},
            {"s": 0.0, "lateral": 3.5, "speed": 1.0, "trigger_speed": 6.0}]},
        duration=45.0,
        check=lambda r, p: _expect(r, [
            ("collisions", r["collisions"] == 0),
        ] + ([] if p == "road_graph" else [
            ("aborted", r["aborts"] >= 1),
            ("bumper gap to the lead ≥ 2 m", (r["min_lead_gap_m"] or 99) >= 2.0),
            ("body clearance to the passing car ≥ 0.3 m",
             (r["min_side_clearance_m"] if r["min_side_clearance_m"] is not None else 9) >= 0.3),
        ])),
    ),
    # Slow lead, but the neighbour lane is oncoming: follow, never change.
    "no_oncoming": dict(
        scenario={"lanes": "oncoming", "obstacles": [{"s": 25.0, "lateral": 0.0, "speed": 1.0}]},
        duration=35.0,
        check=lambda r, p: _expect(r, [
            ("collisions", r["collisions"] == 0),
            ("no lane change", r["lane_changes"] == 0 and r["max_lateral_m"] < 1.0),
            ("kept following", r["final_speed_mps"] > 0.5),
        ]),
    ),
    # Goal on the curve: route there and stop on it
    "goal": dict(
        scenario={}, duration=35.0, goal_s=60.0,
        check=lambda r, p: _expect(r, [
            ("collisions", r["collisions"] == 0),
            ("stopped", r["final_speed_mps"] < 0.1),
            ("stopped within 2 m of the goal", abs(r["goal_error_m"]) <= 2.0),
        ]),
    ),
}


def _expect(r, checks):
    return [name for name, ok in checks if not ok]


def _goal_args(goal_s, scenario):
    """planner.launch.py goal_x/goal_y for a goal at arc length goal_s."""
    if "radius" in scenario:
        raise ValueError("goal scenarios assume the default road radius")
    sys.path.insert(0, str(HERE / "fake_carla"))
    import offline_road   # default radius: OFFLINE_SIM_SCENARIO is unset in this process
    x, y = offline_road.point_at(goal_s, 0.0)
    return [f"goal_x:={x:.3f}", f"goal_y:={y:.3f}"]


def run_one(planner, scenario, workdir, verbose, spec=None, rate=1.0, domain=None):
    spec = spec or SCENARIOS[scenario]
    duration, check, goal_s = spec["duration"], spec["check"], spec.get("goal_s")
    start = spec.get("start", {})
    scen_file = workdir / f"{scenario}.json"
    scen_file.write_text(json.dumps(spec["scenario"]))
    clock_file = workdir / f"{planner}_{scenario}.clock"
    clock_file.unlink(missing_ok=True)
    trigger_file = workdir / f"{planner}_{scenario}.trigger"
    trigger_file.unlink(missing_ok=True)
    out = workdir / f"{planner}_{scenario}.json"
    log = open(workdir / f"{planner}_{scenario}.log", "w")

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{HERE / 'fake_carla'}:{env.get('PYTHONPATH', '')}"
    env["OFFLINE_SIM_SCENARIO"] = str(scen_file)
    env["OFFLINE_SIM_CLOCK"] = str(clock_file)
    env["OFFLINE_SIM_TRIGGER"] = str(trigger_file)
    env["ROS_DOMAIN_ID"] = str(domain) if domain is not None else \
        env.get("OFFLINE_SIM_DOMAIN_ID", "77")
    env["PYTHONUNBUFFERED"] = "1"

    cmds = [
        ["ros2", "launch", "av_perception", "perception.launch.py"],
        ["ros2", "launch", "av_planning", "planner.launch.py", f"planner_type:={planner}"]
        + (_goal_args(goal_s, spec["scenario"]) if goal_s is not None else []),
        ["ros2", "launch", "av_control", "controller.launch.py"],
    ]
    procs = [subprocess.Popen(c, env=env, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) for c in cmds]
    try:
        sim = subprocess.run(
            [sys.executable, str(HERE / "sim_node.py"), "--duration", str(duration),
             "--out", str(out), "--rate", str(rate),
             "--start-s", str(start.get("s", 5.0)),
             "--start-lateral", str(start.get("lateral", 0.0)),
             "--start-yaw-deg", str(start.get("yaw_deg", 0.0))]
            + (["--goal-s", str(goal_s)] if goal_s is not None else []),
            env=env, stdout=log, stderr=subprocess.STDOUT, timeout=duration / rate + 90,
        )
    finally:
        for p in procs:
            p.send_signal(signal.SIGINT)   # launch forwards it to its nodes
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        log.close()

    if sim.returncode != 0 or not out.exists():
        return None, [f"sim_node failed (see {workdir / f'{planner}_{scenario}.log'})"]
    result = json.loads(out.read_text())
    if result["trajectory_msgs"] == 0:
        return result, ["no trajectory received — stack did not come up"]
    return result, check(result, planner)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--planners", nargs="+", default=["road_graph", "lattice", "mppi"])
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--workdir", type=Path, default=None, help="keep logs/results here")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="simulated seconds per wall second (the stack follows /clock)")
    ap.add_argument("--sweep", type=int, default=0, metavar="N",
                    help="run N randomized variants (sweep.py) instead of the named scenarios")
    ap.add_argument("--seed", type=int, default=0, help="sweep seed")
    ap.add_argument("--replay", type=Path, default=None,
                    help="re-run the variants in a sweep's failures.json")
    ap.add_argument("--jobs", type=int, default=1,
                    help="runs in parallel (each on its own ROS domain)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="offline_sim_"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"Logs and results: {workdir}\n")

    if args.sweep or args.replay:
        sys.exit(run_sweep(args, workdir))

    failures = 0
    rows = []
    for planner in args.planners:
        for scenario in args.scenarios:
            t0 = time.time()
            print(f"▶ {planner:<10} {scenario:<8} …", end="", flush=True)
            result, failed = run_one(planner, scenario, workdir, args.verbose, rate=args.rate)
            status = "PASS" if not failed else "FAIL: " + "; ".join(failed)
            failures += bool(failed)
            print(f" {status}  ({time.time() - t0:.0f} s)")
            if result:
                rows.append((planner, scenario, result))
                if args.verbose or failed:
                    print("   " + json.dumps(result))

    print("\nplanner     scenario  progress  max|CTE|  min_obs  coll  lane_err  final_v  gap/goal")
    for planner, scenario, r in rows:
        mo = r["min_obstacle_dist_m"]
        le = r["lane_offset_mean_abs_err_m"]
        extra = ""
        if r.get("goal_error_m") is not None:
            extra = f"goal {r['goal_error_m']:+.1f}m"
        elif r.get("min_lead_gap_m") is not None:
            extra = f"gap {r['min_lead_gap_m']:.1f}m"
        print(f"{planner:<11} {scenario:<9} {r['final_s_m']:>7.1f}m {r['max_abs_cte_m']:>7.2f}m "
              f"{'   -   ' if mo is None else f'{mo:>6.1f}m'} {r['collisions']:>4}  "
              f"{'   -   ' if le is None else f'{le:>6.2f}m'}  {r['final_speed_mps']:>5.2f}  {extra}")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)


def run_sweep(args, workdir) -> int:
    """Run sweep variants (or replay failures); report pass rates. Returns exit status."""
    if args.replay:
        entries = json.loads(args.replay.read_text())
        jobs = [(e["planner"], e["params"]) for e in entries]
        print(f"Replaying {len(jobs)} failed run(s) from {args.replay}\n")
    else:
        variants = [sweep.sample(args.seed, i) for i in range(args.sweep)]
        jobs = [(planner, p) for p in variants for planner in args.planners]
        print(f"Sweep: {args.sweep} variants (seed {args.seed}) × {len(args.planners)} planners"
              f" = {len(jobs)} runs, rate {args.rate}×, {args.jobs} in parallel\n")

    base_domain = int(os.environ.get("OFFLINE_SIM_DOMAIN_ID", "77"))
    free_domains = list(range(base_domain, base_domain + max(1, args.jobs)))

    def work(job):
        planner, params = job
        domain = free_domains.pop()
        try:
            t0 = time.time()
            result, failed = run_one(planner, sweep.name(params), workdir, args.verbose,
                                     spec=sweep.build(params), rate=args.rate, domain=domain)
            return planner, params, result, failed, time.time() - t0
        finally:
            free_domains.append(domain)

    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for planner, params, result, failed, dt in pool.map(work, jobs):
            outcomes.append((planner, params, result, failed))
            status = "PASS" if not failed else "FAIL: " + "; ".join(failed)
            print(f"▶ {planner:<10} {sweep.name(params):<22} {status}  ({dt:.0f} s)", flush=True)
            if failed or args.verbose:
                print("   params " + json.dumps(params))
                if result:
                    print("   " + json.dumps(result))

    # pass rates: family × planner
    stats = {}
    for planner, params, _, failed in outcomes:
        for key in ((params["family"], planner), (params["family"], "all"), ("all", planner),
                    ("all", "all")):
            n, ok = stats.get(key, (0, 0))
            stats[key] = (n + 1, ok + (not failed))
    planners = sorted({pl for pl, *_ in outcomes}) + ["all"]
    families = [f for f in sweep.FAMILIES if (f, "all") in stats] + ["all"]
    print("\npass rate   " + "".join(f"{pl:>14}" for pl in planners))
    for fam in families:
        cells = []
        for pl in planners:
            n, ok = stats.get((fam, pl), (0, 0))
            cells.append(f"{ok:>4}/{n:<3} {100 * ok / n:>4.0f}%" if n else f"{'-':>14}")
        print(f"{fam:<11} " + "".join(f"{c:>14}" for c in cells))

    failed_runs = [{"planner": pl, "params": params, "failures": failed}
                   for pl, params, _, failed in outcomes if failed]
    summary = {"seed": args.seed, "variants": args.sweep, "rate": args.rate,
               "pass_rates": {f"{k[0]}/{k[1]}": {"runs": n, "passed": ok}
                              for k, (n, ok) in stats.items()}}
    (workdir / "sweep_summary.json").write_text(json.dumps(summary, indent=1))
    if failed_runs:
        path = workdir / "failures.json"
        path.write_text(json.dumps(failed_runs, indent=1))
        print(f"\n{len(failed_runs)} FAILED — replay with:  "
              f"python3 scripts/offline_sim/run_scenarios.py --replay {path} -v")
        return 1
    print("\nALL PASSED")
    return 0


if __name__ == "__main__":
    main()
