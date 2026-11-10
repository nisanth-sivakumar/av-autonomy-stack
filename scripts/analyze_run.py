#!/usr/bin/env python3
"""
scripts/analyze_run.py
-----------------------
Parses a ROS2 bag recorded by reset_run.sh and produces:
  1. run_data.csv   — timestamped ego state, control, CTE, perception
  2. metrics.txt    — summary metrics
  3. run_plots.png  — driven vs planned path, speed, CTE + perceived lane
                      offset, steering, throttle/brake, obstacle distance

Cross-track error is measured against the *planned trajectory* that was
active at each instant (/planning/trajectory), not against the driven path.
Bags without that topic (pre-planner runs) get no CTE.

Usage
-----
    python3 scripts/analyze_run.py bags/run_mppi_20260509_032717
    python3 scripts/analyze_run.py bags/<run> -o out_dir --no-show

Requirements
------------
    pip3 install --user rosbags matplotlib pandas "numpy<2"
    (rosbags is a pure-Python bag reader — no ROS installation needed.)
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.gridspec as gridspec   # noqa: E402

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
except ImportError:
    print("ERROR: rosbags not installed.")
    print("  pip3 install --user rosbags")
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent
CARLA_MSGS_DIR = REPO / "carla_ros_ws/src/ros-bridge/carla_msgs/msg"

ODOM_TOPIC = "/carla/ego_vehicle/odometry"
CONTROL_TOPIC = "/carla/ego_vehicle/vehicle_control_cmd"
SPEED_TOPIC = "/carla/ego_vehicle/speedometer"
COLLISION_TOPIC = "/carla/ego_vehicle/collision"
LANE_INV_TOPIC = "/carla/ego_vehicle/lane_invasion"
TRAJ_TOPIC = "/planning/trajectory"
OBST_TOPIC = "/perception/obstacles"
LANE_OFF_TOPIC = "/perception/lane_offset"
BEHAVIOR_TOPIC = "/planning/behavior"
COMPUTE_TOPIC = "/control/compute_time_ms"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quat_to_yaw(x, y, z, w) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def signed_cte(px, py, path_xy: np.ndarray) -> float:
    """
    Signed distance from (px, py) to a polyline (+ = left of path).

    The first segment is extended 20 m backwards: planned paths start a
    few metres ahead of the vehicle, and without the extension the
    closest point would be the path's first point, mixing longitudinal
    distance into the CTE.
    """
    if len(path_xy) < 2:
        return float("nan")
    d0 = path_xy[1] - path_xy[0]
    n0 = np.linalg.norm(d0)
    if n0 < 1e-9:
        return float("nan")
    pts = np.vstack([path_xy[0] - d0 / n0 * 20.0, path_xy])
    a, b = pts[:-1], pts[1:]
    ab = b - a
    ab_len2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    p = np.array([px, py])
    t = np.clip(((p - a) * ab).sum(axis=1) / ab_len2, 0.0, 1.0)
    closest = a + t[:, None] * ab
    dist = np.linalg.norm(p - closest, axis=1)
    i = int(np.argmin(dist))
    cross = ab[i, 0] * (py - a[i, 1]) - ab[i, 1] * (px - a[i, 0])
    return float(math.copysign(dist[i], cross))


def covered_cte(px, py, path_xy: np.ndarray, max_dist: float = 3.0):
    """
    Signed distance to the path if the car is *beside* it (its closest
    point is not the path's first vertex from behind), else None.
    """
    if len(path_xy) < 2:
        return None
    a, b = path_xy[:-1], path_xy[1:]
    ab = b - a
    ab_len2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    p = np.array([px, py])
    t_raw = ((p - a) * ab).sum(axis=1) / ab_len2
    t = np.clip(t_raw, 0.0, 1.0)
    dist = np.linalg.norm(p - (a + t[:, None] * ab), axis=1)
    i = int(np.argmin(dist))
    if (i == 0 and t_raw[0] < 0.0) or dist[i] > max_dist:
        return None
    cross = ab[i, 0] * (py - a[i, 1]) - ab[i, 1] * (px - a[i, 0])
    return float(math.copysign(dist[i], cross))


def compute_jerk(steer_series, timestamps):
    """RMS of the steering-command rate (1/s). Lower = smoother."""
    if len(steer_series) < 2:
        return 0.0
    dt = np.maximum(np.diff(timestamps), 1e-6)
    return float(np.sqrt(np.mean((np.diff(steer_series) / dt) ** 2)))


def count_events(times, gap_s=1.0):
    """Number of distinct events: messages separated by more than gap_s."""
    if not times:
        return 0
    t = np.sort(np.asarray(times))
    return int(1 + np.sum(np.diff(t) > gap_s))


DERIVED_MSGS_DIR = Path("/opt/ros/humble/share/derived_object_msgs/msg")


def behavior_metrics(samples) -> dict:
    """Lane changes, aborts and the state sequence from /planning/behavior."""
    if not samples:
        return {"Lane changes / aborts": "n/a (no /planning/behavior)"}
    seq = []
    for _, state in samples:
        if not seq or seq[-1] != state:
            seq.append(state)
    changes = sum(1 for a, b in zip(seq, seq[1:]) if a == "CHANGING" and b != "ABORT")
    aborts = seq.count("ABORT")
    shown = " > ".join(seq[:8]) + (" > ..." if len(seq) > 8 else "")
    return {"Lane changes / aborts": f"{changes} / {aborts}", "Behavior states": shown}


def make_typestore():
    """Humble types plus carla_msgs (ros-bridge) and derived_object_msgs (/perception/objects)."""
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    for pkg, msg_dir in (("carla_msgs", CARLA_MSGS_DIR), ("derived_object_msgs", DERIVED_MSGS_DIR)):
        if not msg_dir.is_dir():
            print(f"  WARNING: {msg_dir} not found — {pkg} topics unreadable")
            continue
        types = {}
        for f in sorted(msg_dir.glob("*.msg")):
            types.update(get_types_from_msg(f.read_text(), f"{pkg}/msg/{f.stem}"))
        typestore.register(types)
    return typestore


# ---------------------------------------------------------------------------
# Bag reading
# ---------------------------------------------------------------------------

def read_bag(bag_path: Path) -> dict:
    typestore = make_typestore()
    data = {k: [] for k in ("odom", "control", "speed", "traj", "obst", "lane",
                            "collision", "lane_inv", "behavior", "compute")}
    print(f"Reading bag: {bag_path}")

    with Reader(bag_path) as reader:
        topics = {c.topic for c in reader.connections}
        for topic in (ODOM_TOPIC, CONTROL_TOPIC, TRAJ_TOPIC):
            if topic not in topics:
                print(f"  WARNING: topic {topic} not found in bag")

        for conn, timestamp, raw in reader.messages():
            t = timestamp * 1e-9
            topic = conn.topic
            if topic not in (ODOM_TOPIC, CONTROL_TOPIC, SPEED_TOPIC, TRAJ_TOPIC, OBST_TOPIC,
                             LANE_OFF_TOPIC, COLLISION_TOPIC, LANE_INV_TOPIC, BEHAVIOR_TOPIC,
                             COMPUTE_TOPIC):
                continue
            try:
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
            except Exception as exc:   # unknown type etc.
                print(f"  WARNING: cannot decode {topic} ({conn.msgtype}): {exc}")
                topics.discard(topic)
                continue

            if topic == ODOM_TOPIC:
                p, q, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear
                data["odom"].append({
                    "t": t, "t_sim": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    "x": p.x, "y": p.y,
                    "yaw_deg": math.degrees(quat_to_yaw(q.x, q.y, q.z, q.w)),
                    "speed": math.hypot(v.x, v.y),
                })
            elif topic == CONTROL_TOPIC:
                data["control"].append({"t": t, "throttle": msg.throttle,
                                        "steer": msg.steer, "brake": msg.brake})
            elif topic == SPEED_TOPIC:
                data["speed"].append({"t": t, "speed_sensor": msg.data})   # std_msgs/Float32
            elif topic == TRAJ_TOPIC:
                xy = np.array([[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses])
                data["traj"].append((t, xy.reshape(-1, 2)))
            elif topic == OBST_TOPIC:
                data["obst"].append((t, [(p.position.x, p.position.y) for p in msg.poses]))
            elif topic == LANE_OFF_TOPIC:
                data["lane"].append({"t": t, "lane_offset": msg.data})
            elif topic == COLLISION_TOPIC:
                data["collision"].append(t)
            elif topic == LANE_INV_TOPIC:
                data["lane_inv"].append(t)
            elif topic == BEHAVIOR_TOPIC:
                data["behavior"].append((t, msg.data.split()[0].split("=", 1)[-1]))
            elif topic == COMPUTE_TOPIC:
                data["compute"].append(msg.data)

    for k in ("odom", "control", "speed", "traj", "obst", "lane", "collision", "lane_inv",
              "behavior", "compute"):
        print(f"  {k:<10}: {len(data[k])} msgs")
    return data


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(bag_path: Path, output_dir: Path, show: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = read_bag(bag_path)
    if not data["odom"]:
        print("ERROR: No odometry data in bag. Was the bridge running?")
        sys.exit(1)

    odom = pd.DataFrame(data["odom"]).sort_values("t")
    # CARLA briefly reports the ego at (0, 0) while spawning — drop it
    odom = odom[~((odom["x"].abs() < 0.1) & (odom["y"].abs() < 0.1))].reset_index(drop=True)
    t0 = odom["t"].iloc[0]

    # Odometry that teleports back and forth means two actors with
    # role_name "ego_vehicle" exist (a previous run's ego was not destroyed).
    step = np.hypot(np.diff(odom["x"]), np.diff(odom["y"]))
    dt = np.maximum(np.diff(odom["t"]), 1e-3)
    jumps = int(np.sum(step > 1.0 + 3.0 * odom["speed"].to_numpy()[1:] * dt))
    if jumps > 5:
        print(f"\n  WARNING: {jumps} odometry jumps — more than one 'ego_vehicle' was publishing.\n"
              "  Metrics below mix both vehicles. Restart CARLA or destroy the stale actor.\n")

    # ── CTE vs the planned trajectory at each odom sample ─────────────────
    # Plans start up to one waypoint (3 m) ahead of the car.  Extrapolating
    # the first segment backwards misreads the car as being inside the
    # curve in tight turns, so use the most recent plan that passes beside
    # the car (plans overlap from tick to tick), falling back to the
    # extrapolation only if none does.
    cte = np.full(len(odom), np.nan)
    if data["traj"]:
        traj_t = np.array([t for t, _ in data["traj"]])
        for i, row in enumerate(odom.itertuples()):
            j = int(np.searchsorted(traj_t, row.t, side="right")) - 1
            if j < 0:
                continue
            for k in range(j, max(-1, j - 50), -1):
                c = covered_cte(row.x, row.y, data["traj"][k][1])
                if c is not None:
                    cte[i] = c
                    break
            else:
                if len(data["traj"][j][1]) >= 2:
                    cte[i] = signed_cte(row.x, row.y, data["traj"][j][1])
    odom["cte"] = cte

    # ── Nearest perceived obstacle at each odom sample ────────────────────
    obs_dist = np.full(len(odom), np.nan)
    if data["obst"]:
        obs_t = np.array([t for t, _ in data["obst"]])
        for i, row in enumerate(odom.itertuples()):
            j = int(np.searchsorted(obs_t, row.t, side="right")) - 1
            if j >= 0 and data["obst"][j][1]:
                obs_dist[i] = min(math.hypot(ox - row.x, oy - row.y) for ox, oy in data["obst"][j][1])
    odom["nearest_obstacle"] = obs_dist

    # ── Merge everything on the odom timeline ─────────────────────────────
    merged = odom.copy()
    for key in ("control", "speed", "lane"):
        if data[key]:
            df = pd.DataFrame(data[key]).sort_values("t")
            merged = pd.merge_asof(merged, df, on="t", direction="nearest", tolerance=0.5)
    for col in ("throttle", "steer", "brake", "lane_offset"):
        if col not in merged:
            merged[col] = np.nan
    merged["t"] -= t0
    odom["t"] -= t0

    csv_path = output_dir / "run_data.csv"
    merged.to_csv(csv_path, index=False)
    print(f"CSV saved → {csv_path}")

    # ── Metrics ───────────────────────────────────────────────────────────
    total_dist = float(np.sum(np.hypot(np.diff(odom["x"]), np.diff(odom["y"]))))
    ctl = merged.dropna(subset=["steer"])
    abs_cte = np.abs(odom["cte"].dropna())
    lane = merged["lane_offset"].dropna()

    # bag stamps are wall time; CARLA runs faster than real time in sync mode
    wall = float(odom["t"].iloc[-1] - odom["t"].iloc[0])
    sim = float(odom["t_sim"].iloc[-1] - odom["t_sim"].iloc[0])
    sim_per_wall = sim / wall if wall > 0 and sim > 0 else 1.0
    moving = odom[odom["speed"] > 0.5]
    jerk = None
    if len(moving) > 3:
        ts = moving["t_sim"].to_numpy()
        acc = np.gradient(moving["speed"].to_numpy(), ts)
        jerk = float(np.sqrt(np.mean(np.gradient(acc, ts) ** 2)))
    comp = np.asarray(data["compute"], float)
    # Tracking: skip the first 3 s of motion, where the error is the spawn
    # point's offset from the lane (0.6 m at the default spawn), not tracking,
    # and stop at the controller's last command (the car rolls on after it
    # stops until the stop script's brake, or, in older bags, the ego is gone)
    track_cte = None
    if len(moving):
        t_go = float(moving["t_sim"].iloc[0])
        # last command that isn't a full brake (reset_run.sh brakes at the end)
        drive = ctl[ctl["brake"] < 0.99]
        t_end = float(drive["t"].max()) if len(drive) else float(moving["t"].max())
        win = moving[(moving["t_sim"] > t_go + 3.0) & (moving["t"] <= t_end)]
        tc = np.abs(win["cte"].dropna())
        track_cte = float(tc.max()) if len(tc) else None

    def fmt(v, spec=".3f", unit=""):
        return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:{spec}}{unit}"

    metrics = {
        "Odometry jumps (multi-ego)":    str(jumps),
        "Run duration (s)":              fmt(float(odom["t"].iloc[-1]), ".1f"),
        # the stack runs on simulated time; CARLA may run faster than real time
        "Simulated duration (s)":        fmt(float(odom["t_sim"].iloc[-1] - odom["t_sim"].iloc[0]), ".1f"),
        "Distance driven (m)":           fmt(total_dist, ".1f"),
        "Mean speed (m/s)":              fmt(float(odom["speed"].mean()), ".2f"),
        "Max speed (m/s)":               fmt(float(odom["speed"].max()), ".2f"),
        "Mean |CTE| vs plan (m)":        fmt(float(abs_cte.mean()) if len(abs_cte) else None),
        "Max |CTE| vs plan (m)":         fmt(float(abs_cte.max()) if len(abs_cte) else None),
        "Max |CTE| tracking (m)":        fmt(track_cte),   # moving, after the start
        # steer command units (normalized by 70 deg) per simulated second
        "Steering rate RMS (1/s, sim)":  fmt(compute_jerk(ctl["steer"].values, ctl["t"].values)
                                             / sim_per_wall, ".4f"),
        "Longitudinal jerk RMS (m/s3)":  fmt(jerk, ".2f"),
        "Controller compute p50/p99 (ms)": (f"{np.percentile(comp, 50):.3f} / "
                                            f"{np.percentile(comp, 99):.3f}" if len(comp) else
                                            "n/a (no /control/compute_time_ms)"),
        "Braking time (%)":              fmt(float((ctl["brake"] > 0.1).mean() * 100) if len(ctl) else None, ".1f"),
        "Collisions (events)":           str(count_events(data["collision"])),
        "Lane invasions (events)":       str(count_events(data["lane_inv"])),
        **behavior_metrics(data["behavior"]),
        "Min perceived obstacle (m)":    fmt(float(np.nanmin(obs_dist)) if np.isfinite(obs_dist).any() else None, ".2f"),
        "Mean |lane offset| (m)":        fmt(float(lane.abs().mean()) if len(lane) else None),
        # share of odom samples with a lane-offset message within 0.5 s
        "Lane offset availability (%)":  fmt(float(merged["lane_offset"].notna().mean() * 100)
                                             if data["lane"] else None, ".0f"),
    }

    lines = ["RUN METRICS SUMMARY", "=" * 46]
    lines += [f"  {k:<30} {v}" for k, v in metrics.items()]
    lines += ["=" * 46]
    print("\n" + "\n".join(lines) + "\n")
    metrics_path = output_dir / "metrics.txt"
    metrics_path.write_text("\n".join(lines) + "\n")
    print(f"Metrics saved → {metrics_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    # numpy arrays: the system matplotlib cannot index newer pandas Series
    O = {c: odom[c].to_numpy(dtype=float) for c in odom.columns}
    M = {c: merged[c].to_numpy(dtype=float) for c in merged.columns}
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Run Analysis — {bag_path.name}", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    ax = fig.add_subplot(gs[:, 0])
    ax.plot(O["x"], O["y"], lw=1.5, color="royalblue", label="Driven path")
    if data["traj"]:
        for _, xy in data["traj"][:: max(1, len(data["traj"]) // 15)]:
            if len(xy):
                ax.plot(xy[:, 0], xy[:, 1], lw=0.6, color="gray", alpha=0.4)
        ax.plot([], [], lw=0.6, color="gray", label="Planned (sampled)")
    obs_pts = [p for _, ps in data["obst"] for p in ps]
    if obs_pts:
        op = np.asarray(obs_pts)
        ax.scatter(op[:, 0], op[:, 1], s=2, color="red", alpha=0.2, label="Perceived obstacles")
    ax.plot(O["x"][0], O["y"][0], "go", ms=8, label="Start")
    ax.plot(O["x"][-1], O["y"][-1], "rs", ms=8, label="End")
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="Trajectory")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(O["t"], O["speed"], color="darkorange", lw=1.2)
    ax.set(xlabel="Time (s)", ylabel="Speed (m/s)", title="Speed")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(O["t"], O["cte"], color="crimson", lw=1.2, label="CTE vs plan (+ left)")
    if len(lane):
        ax.plot(M["t"], M["lane_offset"], color="teal", lw=1.0, alpha=0.8,
                label="Perceived lane offset (+ left)")
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set(xlabel="Time (s)", ylabel="m", title="Lateral error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 3])
    ax.plot(O["t"], O["nearest_obstacle"], color="red", lw=1.0)
    for tc in data["collision"]:
        ax.axvline(tc - t0, color="black", lw=0.5, alpha=0.3)
    ax.set(xlabel="Time (s)", ylabel="m", title="Nearest perceived obstacle (lines = collisions)")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(M["t"], M["steer"], color="purple", lw=1.2)
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set(xlabel="Time (s)", ylabel="Steer [-1, 1] (+ right)", title="Steering command")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(M["t"], M["throttle"], color="green", lw=1.2, label="Throttle")
    ax.plot(M["t"], M["brake"], color="red", lw=1.2, label="Brake")
    ax.set(xlabel="Time (s)", ylabel="Command [0, 1]", title="Throttle / Brake")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    if len(abs_cte):
        ax.hist(O["cte"][~np.isnan(O["cte"])], bins=40, color="crimson", alpha=0.7)
    ax.set(xlabel="CTE (m)", ylabel="samples", title="CTE distribution")
    ax.grid(True, alpha=0.3)

    plot_path = output_dir / "run_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plots saved → {plot_path}")
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a ROS2 bag recorded by reset_run.sh.")
    parser.add_argument("bag_path", type=Path, help="Bag directory, e.g. bags/run_mppi_<timestamp>")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: the bag directory)")
    parser.add_argument("--no-show", action="store_true", help="Don't open the plot window")
    args = parser.parse_args()
    analyze(args.bag_path.resolve(), (args.output or args.bag_path).resolve(), not args.no_show)
