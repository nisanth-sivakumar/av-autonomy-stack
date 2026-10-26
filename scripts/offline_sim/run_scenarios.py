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
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name → (obstacles, duration_s, checks(result, planner) → list of failures)
SCENARIOS = {
    # Straight → 90° left curve, empty road: lane keeping + perception accuracy
    "curve": (
        [], 35.0,
        lambda r, p: _expect(r, [
            ("collisions", r["collisions"] == 0),
            ("progress > 75 m", r["final_s_m"] > 75),
            ("max |CTE| < 0.6 m", r["max_abs_cte_m"] < 0.6),
            ("lane-offset msgs received", r["lane_offset_msgs"] > 50),
            ("lane-offset mean err < 0.3 m", (r["lane_offset_mean_abs_err_m"] or 9) < 0.3),
        ]),
    ),
    # Car parked in the middle of the lane on the straight: every planner must stop
    "blocked": (
        [{"s": 28.0, "lateral": 0.0}], 25.0,
        lambda r, p: _expect(r, [
            ("collisions", r["collisions"] == 0),
            ("stopped", r["final_speed_mps"] < 0.1),
            ("stopped ≥ 4.5 m short (centres)", r["min_obstacle_dist_m"] >= 4.5),
            ("stopped ≤ 12 m short", r["min_obstacle_dist_m"] <= 12.0),
        ]),
    ),
    # Car parked 1.0 m left of centre on the curve: lattice/MPPI pass it,
    # road_graph (no avoidance) must stop.
    "partial": (
        [{"s": 50.0, "lateral": 1.0}], 40.0,
        lambda r, p: _expect(r, [
            ("collisions", r["collisions"] == 0),
        ] + ([
            ("stopped", r["final_speed_mps"] < 0.1),
        ] if p == "road_graph" else [
            ("passed the obstacle", r["final_s_m"] > 60),
        ])),
    ),
}


def _expect(r, checks):
    return [name for name, ok in checks if not ok]


def run_one(planner, scenario, workdir, verbose):
    obstacles, duration, check = SCENARIOS[scenario]
    scen_file = workdir / f"{scenario}.json"
    scen_file.write_text(json.dumps({"obstacles": obstacles}))
    out = workdir / f"{planner}_{scenario}.json"
    log = open(workdir / f"{planner}_{scenario}.log", "w")

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{HERE / 'fake_carla'}:{env.get('PYTHONPATH', '')}"
    env["OFFLINE_SIM_SCENARIO"] = str(scen_file)
    env["ROS_DOMAIN_ID"] = env.get("OFFLINE_SIM_DOMAIN_ID", "77")
    env["PYTHONUNBUFFERED"] = "1"

    cmds = [
        ["ros2", "launch", "av_perception", "perception.launch.py"],
        ["ros2", "launch", "av_planning", "planner.launch.py", f"planner_type:={planner}"],
        ["ros2", "launch", "av_control", "controller.launch.py"],
    ]
    procs = [subprocess.Popen(c, env=env, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) for c in cmds]
    try:
        sim = subprocess.run(
            [sys.executable, str(HERE / "sim_node.py"), "--duration", str(duration),
             "--out", str(out)],
            env=env, stdout=log, stderr=subprocess.STDOUT, timeout=duration + 60,
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
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="offline_sim_"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"Logs and results: {workdir}\n")

    failures = 0
    rows = []
    for planner in args.planners:
        for scenario in args.scenarios:
            t0 = time.time()
            print(f"▶ {planner:<10} {scenario:<8} …", end="", flush=True)
            result, failed = run_one(planner, scenario, workdir, args.verbose)
            status = "PASS" if not failed else "FAIL: " + "; ".join(failed)
            failures += bool(failed)
            print(f" {status}  ({time.time() - t0:.0f} s)")
            if result:
                rows.append((planner, scenario, result))
                if args.verbose or failed:
                    print("   " + json.dumps(result))

    print("\nplanner     scenario  progress  max|CTE|  min_obs  coll  lane_err  final_v")
    for planner, scenario, r in rows:
        mo = r["min_obstacle_dist_m"]
        le = r["lane_offset_mean_abs_err_m"]
        print(f"{planner:<11} {scenario:<9} {r['final_s_m']:>7.1f}m {r['max_abs_cte_m']:>7.2f}m "
              f"{'   -   ' if mo is None else f'{mo:>6.1f}m'} {r['collisions']:>4}  "
              f"{'   -   ' if le is None else f'{le:>6.2f}m'}  {r['final_speed_mps']:>5.2f}")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
