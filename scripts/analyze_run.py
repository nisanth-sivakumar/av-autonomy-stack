#!/usr/bin/env python3
"""
scripts/analyze_run.py
-----------------------
Parses a ROS2 bag from a Pure Pursuit run and produces:
  1. A CSV of timestamped ego state + control signals
  2. Cross-track error vs time
  3. Speed vs time
  4. Trajectory plot (driven path vs reference waypoints)
  5. A summary table of key metrics

Usage
-----
    python3 scripts/analyze_run.py bags/run_20240428_120000

Requirements
------------
    pip install rosbags matplotlib pandas numpy

    rosbags is the pure-Python ROS2 bag reader — no ROS installation needed.
    Install it in WSL:
        pip3 install rosbags --break-system-packages
"""

import sys
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    print("ERROR: rosbags not installed.")
    print("  pip3 install rosbags --break-system-packages")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quat_to_yaw(x, y, z, w) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def cross_track_error(px, py, waypoints):
    """
    Compute the minimum distance from point (px, py) to the nearest
    segment of the reference path defined by waypoints.
    Returns a signed CTE: positive = left of path, negative = right.
    """
    if len(waypoints) < 2:
        return 0.0

    min_dist = float('inf')
    sign = 1.0

    for i in range(len(waypoints) - 1):
        ax, ay = waypoints[i]
        bx, by = waypoints[i + 1]

        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay

        ab_len_sq = abx ** 2 + aby ** 2
        if ab_len_sq < 1e-9:
            continue

        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
        closest_x = ax + t * abx
        closest_y = ay + t * aby

        dist = math.hypot(px - closest_x, py - closest_y)
        if dist < min_dist:
            min_dist = dist
            # Sign: cross product of AB and AP
            cross = abx * apy - aby * apx
            sign = 1.0 if cross >= 0 else -1.0

    return sign * min_dist


def compute_jerk(throttle_series, steer_series, timestamps):
    """
    Jerk proxy: RMS of the first derivative of steering angle.
    Lower = smoother driving.
    """
    if len(steer_series) < 2:
        return 0.0
    dt = np.diff(timestamps)
    dt = np.where(dt < 1e-6, 1e-6, dt)
    dsteer = np.diff(steer_series) / dt
    return float(np.sqrt(np.mean(dsteer ** 2)))


# ---------------------------------------------------------------------------
# Bag reading
# ---------------------------------------------------------------------------

def read_bag(bag_path: Path):
    """
    Read odometry, control commands, and speedometer from a ROS2 bag.
    Returns three lists of dicts, one per topic.
    """
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    odom_rows    = []
    control_rows = []
    speed_rows   = []

    ODOM_TOPIC    = '/carla/ego_vehicle/odometry'
    CONTROL_TOPIC = '/carla/ego_vehicle/vehicle_control_cmd'
    SPEED_TOPIC   = '/carla/ego_vehicle/speedometer'

    print(f"Reading bag: {bag_path}")

    with Reader(bag_path) as reader:
        connections = {c.topic: c for c in reader.connections}

        for topic in [ODOM_TOPIC, CONTROL_TOPIC, SPEED_TOPIC]:
            if topic not in connections:
                print(f"  WARNING: topic {topic} not found in bag")

        for conn, timestamp, rawdata in reader.messages():
            t_sec = timestamp * 1e-9   # nanoseconds → seconds

            if conn.topic == ODOM_TOPIC:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                p   = msg.pose.pose.position
                q   = msg.pose.pose.orientation
                v   = msg.twist.twist.linear
                yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
                speed = math.hypot(v.x, v.y)
                odom_rows.append({
                    't': t_sec, 'x': p.x, 'y': p.y,
                    'yaw_deg': math.degrees(yaw), 'speed': speed,
                })

            elif conn.topic == CONTROL_TOPIC:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                control_rows.append({
                    't': t_sec,
                    'throttle': msg.throttle,
                    'steer': msg.steer,
                    'brake': msg.brake,
                })

            elif conn.topic == SPEED_TOPIC:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                speed_rows.append({'t': t_sec, 'speed_sensor': msg.speed})

    print(f"  Odometry msgs    : {len(odom_rows)}")
    print(f"  Control msgs     : {len(control_rows)}")
    print(f"  Speedometer msgs : {len(speed_rows)}")

    return odom_rows, control_rows, speed_rows


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(bag_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    odom_rows, control_rows, speed_rows = read_bag(bag_path)

    if not odom_rows:
        print("ERROR: No odometry data in bag. Was the car moving?")
        sys.exit(1)

    odom_df    = pd.DataFrame(odom_rows).sort_values('t').reset_index(drop=True)
    control_df = pd.DataFrame(control_rows).sort_values('t').reset_index(drop=True)

    # Normalise timestamps to start at 0
    t0 = odom_df['t'].iloc[0]
    odom_df['t']    -= t0
    control_df['t'] -= t0

    # ---- Merge control onto odom timeline via nearest-timestamp join ------
    merged = pd.merge_asof(
        odom_df.sort_values('t'),
        control_df.sort_values('t'),
        on='t', direction='nearest'
    )

    # ---- Reference path: the driven trajectory itself (smoothed) ----------
    # In Phase 2 you'll replace this with the planner's output path.
    # For now, downsample the driven path to use as the "reference".
    waypoints = list(zip(odom_df['x'].values[::5], odom_df['y'].values[::5]))

    # ---- Compute cross-track error ----------------------------------------
    print("Computing cross-track error…")
    cte_values = [
        cross_track_error(row.x, row.y, waypoints)
        for row in odom_df.itertuples()
    ]
    odom_df['cte'] = cte_values

    # ---- Save CSV ----------------------------------------------------------
    csv_path = output_dir / 'run_data.csv'
    merged.to_csv(csv_path, index=False)
    print(f"CSV saved → {csv_path}")

    # ---- Metrics summary --------------------------------------------------
    run_duration  = odom_df['t'].iloc[-1]
    total_dist    = sum(
        math.hypot(
            odom_df['x'].iloc[i] - odom_df['x'].iloc[i-1],
            odom_df['y'].iloc[i] - odom_df['y'].iloc[i-1]
        )
        for i in range(1, len(odom_df))
    )
    mean_speed    = odom_df['speed'].mean()
    max_speed     = odom_df['speed'].max()
    mean_cte      = odom_df['cte'].abs().mean()
    max_cte       = odom_df['cte'].abs().max()
    steering_jerk = compute_jerk(
        merged['throttle'].values,
        merged['steer'].values,
        merged['t'].values
    )
    brake_time    = (merged['brake'] > 0.1).sum() / len(merged) * 100

    metrics = {
        'Run duration (s)':       f"{run_duration:.1f}",
        'Distance driven (m)':    f"{total_dist:.1f}",
        'Mean speed (m/s)':       f"{mean_speed:.2f}",
        'Max speed (m/s)':        f"{max_speed:.2f}",
        'Mean |CTE| (m)':         f"{mean_cte:.3f}",
        'Max |CTE| (m)':          f"{max_cte:.3f}",
        'Steering jerk (RMS)':    f"{steering_jerk:.4f}",
        'Braking time (%)':       f"{brake_time:.1f}",
    }

    print("\n" + "=" * 42)
    print("  RUN METRICS SUMMARY")
    print("=" * 42)
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")
    print("=" * 42 + "\n")

    metrics_path = output_dir / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("RUN METRICS SUMMARY\n")
        f.write("=" * 42 + "\n")
        for k, v in metrics.items():
            f.write(f"  {k:<28} {v}\n")
        f.write("=" * 42 + "\n")
    print(f"Metrics saved → {metrics_path}")

    # ---- Plots ------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Run Analysis — {bag_path.name}", fontsize=13, fontweight='bold')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Trajectory
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(odom_df['x'], odom_df['y'], linewidth=1.5, color='royalblue', label='Driven path')
    ax1.plot(odom_df['x'].iloc[0],  odom_df['y'].iloc[0],  'go', markersize=8, label='Start')
    ax1.plot(odom_df['x'].iloc[-1], odom_df['y'].iloc[-1], 'rs', markersize=8, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Trajectory')
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 2. Speed vs time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(odom_df['t'], odom_df['speed'], color='darkorange', linewidth=1.2)
    ax2.axhline(y=merged['throttle'].mean() * 5, color='gray',
                linestyle='--', alpha=0.5, label='Target (approx)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Speed (m/s)')
    ax2.set_title('Speed over Time')
    ax2.grid(True, alpha=0.3)

    # 3. Cross-track error vs time
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(odom_df['t'], odom_df['cte'], color='crimson', linewidth=1.2)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax3.fill_between(odom_df['t'], odom_df['cte'], alpha=0.15, color='crimson')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('CTE (m)')
    ax3.set_title('Cross-Track Error')
    ax3.grid(True, alpha=0.3)

    # 4. Steering vs time
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(merged['t'], merged['steer'], color='purple', linewidth=1.2)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Steer [-1, 1]')
    ax4.set_title('Steering Command')
    ax4.grid(True, alpha=0.3)

    # 5. Throttle and brake vs time
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(merged['t'], merged['throttle'], color='green',  linewidth=1.2, label='Throttle')
    ax5.plot(merged['t'], merged['brake'],    color='red',    linewidth=1.2, label='Brake')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Command [0, 1]')
    ax5.set_title('Throttle / Brake')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    plot_path = output_dir / 'run_plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved → {plot_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze a ROS2 bag from a Pure Pursuit run.'
    )
    parser.add_argument(
        'bag_path',
        type=Path,
        help='Path to the bag directory, e.g. bags/run_20240428_120000'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output directory for CSV and plots (default: same as bag_path)'
    )
    args = parser.parse_args()

    bag_path   = args.bag_path.resolve()
    output_dir = (args.output or args.bag_path).resolve()

    if not bag_path.exists():
        print(f"ERROR: Bag not found at {bag_path}")
        sys.exit(1)

    analyze(bag_path, output_dir)