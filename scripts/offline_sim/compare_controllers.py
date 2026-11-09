#!/usr/bin/env python3
"""
Compare two offline result directories (run_scenarios.py --workdir), e.g.
Pure Pursuit vs MPC, and print markdown tables.

    python3 scripts/offline_sim/compare_controllers.py <pp_dir> <mpc_dir> [--names PP MPC]

Named scenarios: per scenario and planner, pass/fail (the scenario's own
check), max / mean |CTE vs plan|, steering-rate RMS, jerk RMS, compute
time p50/p99 and collisions.  Sweep directories (sweep_summary.json): pass
rates and the same metrics aggregated over every run.

CTE vs plan is the controller's tracking error (distance from the vehicle
centre to the trajectory it was given), so it stays meaningful during
lane changes, unlike CTE vs the road's lane.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_scenarios import SCENARIOS  # noqa: E402

PLANNERS = ("road_graph", "lattice", "mppi")


def load(d: Path, planner: str, scenario: str):
    f = d / f"{planner}_{scenario}.json"
    return json.loads(f.read_text()) if f.exists() else None


def fmt(v, spec=".2f"):
    return "-" if v is None else format(v, spec)


def named(dirs, names):
    scenarios = [s for s in SCENARIOS if any(load(d, p, s) for d in dirs for p in PLANNERS)]
    print("| Scenario | Planner | " + " | ".join(
        f"{n} pass | {n} max/mean CTE (m) | {n} steer rate RMS (rad/s) | {n} jerk RMS (m/s³) | "
        f"{n} compute p50/p99 (ms)" for n in names) + " |")
    print("|---|---|" + "---|---|---|---|---|" * len(names))
    totals = {n: dict(passed=0, runs=0, coll=0, cte_max=[], cte_mean=[], sr=[], jerk=[],
                      p50=[], p99=[]) for n in names}
    for s in scenarios:
        for p in PLANNERS:
            cells = []
            for d, n in zip(dirs, names):
                r = load(d, p, s)
                if r is None:
                    cells += ["-"] * 5
                    continue
                failed = SCENARIOS[s]["check"](r, p) if r.get("trajectory_msgs") else ["no stack"]
                t = totals[n]
                t["runs"] += 1
                t["passed"] += not failed
                t["coll"] += r["collisions"]
                for key, field in (("cte_max", "max_plan_cte_m"), ("cte_mean", "mean_plan_cte_m"),
                                   ("sr", "steer_rate_rms_radps"), ("jerk", "jerk_rms_mps3"),
                                   ("p50", "compute_ms_p50"), ("p99", "compute_ms_p99")):
                    if r.get(field) is not None:
                        t[key].append(r[field])
                cells += ["✓" if not failed else "✗ " + "; ".join(failed),
                          f"{fmt(r.get('max_plan_cte_m'))} / {fmt(r.get('mean_plan_cte_m'), '.3f')}",
                          fmt(r.get("steer_rate_rms_radps"), ".3f"),
                          fmt(r.get("jerk_rms_mps3"), ".2f"),
                          f"{fmt(r.get('compute_ms_p50'), '.3f')} / {fmt(r.get('compute_ms_p99'), '.3f')}"]
            print(f"| {s} | {p} | " + " | ".join(cells) + " |")
    print()
    summary(totals)


def summary(totals):
    print("| Controller | Passed | Collisions | max CTE vs plan (m) | mean CTE (m) | "
          "steer rate RMS (rad/s) | jerk RMS (m/s³) | compute p50 / p99 (ms) |")
    print("|---|---|---|---|---|---|---|---|")
    for n, t in totals.items():
        mx = max(t["cte_max"]) if t["cte_max"] else None
        mean = lambda a: float(np.mean(a)) if a else None  # noqa: E731
        p99 = max(t["p99"]) if t["p99"] else None
        print(f"| {n} | {t['passed']}/{t['runs']} | {t['coll']} | {fmt(mx)} | "
              f"{fmt(mean(t['cte_mean']), '.3f')} | {fmt(mean(t['sr']), '.3f')} | "
              f"{fmt(mean(t['jerk']), '.2f')} | {fmt(mean(t['p50']), '.3f')} / "
              f"{fmt(p99, '.3f')} (worst run) |")


def sweep(dirs, names):
    totals = {}
    for d, n in zip(dirs, names):
        summ = json.loads((d / "sweep_summary.json").read_text())
        rates = summ["pass_rates"]
        t = dict(passed=rates["all/all"]["passed"], runs=rates["all/all"]["runs"], coll=0,
                 cte_max=[], cte_mean=[], sr=[], jerk=[], p50=[], p99=[])
        for f in d.glob("*_s*_*.json"):
            r = json.loads(f.read_text())
            if "collisions" not in r:
                continue
            t["coll"] += r["collisions"]
            for key, field in (("cte_max", "max_plan_cte_m"), ("cte_mean", "mean_plan_cte_m"),
                               ("sr", "steer_rate_rms_radps"), ("jerk", "jerk_rms_mps3"),
                               ("p50", "compute_ms_p50"), ("p99", "compute_ms_p99")):
                if r.get(field) is not None:
                    t[key].append(r[field])
        totals[n] = t
        fams = sorted({k.split("/")[0] for k in rates} - {"all"})
        print(f"{n} (seed {summ['seed']}): " + ", ".join(
            f"{f} {rates[f + '/all']['passed']}/{rates[f + '/all']['runs']}" for f in fams))
    print()
    summary(totals)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--names", nargs="+", default=None)
    args = ap.parse_args()
    names = args.names or [d.name for d in args.dirs]
    if all((d / "sweep_summary.json").exists() for d in args.dirs):
        sweep(args.dirs, names)
    else:
        named(args.dirs, names)


if __name__ == "__main__":
    main()
