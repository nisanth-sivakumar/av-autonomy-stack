#!/usr/bin/env python3
"""
Find places in the current CARLA map for the overtaking scenarios: straight
stretches where the lane has a SAME-direction driving lane on its left, with
no junction for --length metres.

Prints the best candidates (longest, straightest first) with an EGO_SPAWN
value in the ROS frame (y and yaw negated), ready for reset_run.sh and
spawn_scenario.py:

    python3 scripts/find_lane_change_spot.py              # CARLA must be running
    EGO_SPAWN="x,y,z,0,0,yaw" ./scripts/reset_run.sh lattice

Read-only: it spawns nothing.
"""

import argparse
import math

import carla


def same_left(wp):
    """The left neighbour if it is a driving lane in the same direction."""
    nb = wp.get_left_lane()
    if nb is None or nb.lane_type != carla.LaneType.Driving:
        return None
    return nb if (nb.lane_id > 0) == (wp.lane_id > 0) else None


def stretch(wp, length, step=2.0):
    """Walk ahead `length` m; return (ok, max heading change in degrees)."""
    yaw0 = wp.transform.rotation.yaw
    max_dyaw, travelled = 0.0, 0.0
    while travelled < length:
        if wp.is_junction or same_left(wp) is None:
            return False, None
        nxt = wp.next(step)
        if not nxt:
            return False, None
        wp = nxt[0]
        travelled += step
        max_dyaw = max(max_dyaw, abs((wp.transform.rotation.yaw - yaw0 + 180) % 360 - 180))
    return True, max_dyaw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", default="172.25.80.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--length", type=float, default=90.0,
                    help="metres of straight two-lane road needed ahead of the spawn")
    ap.add_argument("--max-turn", type=float, default=8.0, help="max heading change, degrees")
    ap.add_argument("-n", type=int, default=5, help="candidates to print")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    cmap = client.get_world().get_map()
    print(f"Map {cmap.name}: scanning for {args.length:.0f} m two-lane same-direction straights…")

    found, seen = [], set()
    for wp in cmap.generate_waypoints(5.0):
        if wp.lane_type != carla.LaneType.Driving or wp.is_junction:
            continue
        key = (wp.road_id, wp.section_id, wp.lane_id)
        if key in seen or same_left(wp) is None:
            continue
        ok, turn = stretch(wp, args.length)
        if ok and turn <= args.max_turn:
            seen.add(key)      # one candidate per lane segment
            found.append((turn, wp))

    if not found:
        print("None found — try a shorter --length or a larger --max-turn.")
        return
    found.sort(key=lambda c: c[0])
    for turn, wp in found[:args.n]:
        t = wp.transform
        x, y, yaw = t.location.x, -t.location.y, -t.rotation.yaw   # → ROS frame
        print(f"\nroad {wp.road_id} lane {wp.lane_id} (left lane {same_left(wp).lane_id}), "
              f"width {wp.lane_width:.1f} m, heading change {turn:.1f}° over {args.length:.0f} m")
        print(f'  EGO_SPAWN="{x:.2f},{y:.2f},1.5,0.0,0.0,{yaw:.1f}"')


if __name__ == "__main__":
    main()
