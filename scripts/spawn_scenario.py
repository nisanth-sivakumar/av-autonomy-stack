#!/usr/bin/env python3
"""
scripts/spawn_scenario.py
--------------------------
Spawn parked vehicles on the ego's route in CARLA, mirroring the offline
scenarios in scripts/offline_sim/run_scenarios.py:

    blocked : one car in the middle of the ego lane, 30 m ahead
              → every planner should stop ~8 m short (centre-to-centre)
    partial : one car 1.0 m left of the lane centre, 40 m ahead
              → lattice / mppi should pass it, road_graph should stop

The route is walked the same way the planner walks it (waypoint.next(),
first branch at junctions), so the car lands on the lane the ego will drive.

Usage:
    python3 scripts/spawn_scenario.py blocked --keep      # then ./scripts/reset_run.sh <planner>
    python3 scripts/spawn_scenario.py partial --distance 50 --keep
    python3 scripts/spawn_scenario.py custom --distance 35 --lateral -0.8 --keep
    python3 scripts/spawn_scenario.py --clear             # remove scenario cars

Placement reference: the live ego if one exists, otherwise the ego spawn
point in config/objects.json.  The intended flow is therefore: spawn the
scenario with --keep *before* reset_run.sh (which starts driving within
seconds of spawning the ego), run the stack, then --clear.  Scenario cars
have role_name "scenario_obstacle", so reset_run.sh's stale-ego check
ignores them and perception detects them like any other vehicle.

Without --keep the cars stay until Ctrl+C and are destroyed on exit.

Lateral convention: + = LEFT of the lane centre (same as the planners).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

try:
    import carla
except ImportError:
    print("ERROR: CARLA Python API not found (pip3 install --user carla==0.9.15)")
    sys.exit(1)

ROLE = "scenario_obstacle"
OBJECTS_JSON = Path(__file__).resolve().parent.parent / "config" / "objects.json"
SCENARIOS = {
    "blocked": [(30.0, 0.0)],
    "partial": [(40.0, 1.0)],
}


def lateral_location(wp, lateral_left_m: float) -> "carla.Location":
    """Waypoint location shifted `lateral_left_m` to the LEFT of the lane direction.

    CARLA is left-handed (x forward, y right): right = (−sin yaw, cos yaw),
    so a LEFT shift is +L·(sin yaw, −cos yaw).
    """
    loc = wp.transform.location
    yaw = math.radians(wp.transform.rotation.yaw)
    return carla.Location(
        x=loc.x + lateral_left_m * math.sin(yaw),
        y=loc.y - lateral_left_m * math.cos(yaw),
        z=loc.z,
    )


def walk_route(start_wp, distance_m: float, step_m: float = 1.0):
    """Follow the lane `distance_m` ahead like the planner does (first branch)."""
    wp, travelled, junction = start_wp, 0.0, False
    while travelled < distance_m:
        nxt = wp.next(min(step_m, distance_m - travelled))
        if not nxt:
            return None, travelled, junction
        wp = nxt[0]
        travelled += step_m
        junction = junction or getattr(wp, "is_junction", False)
    return wp, travelled, junction


def find_ego(world):
    for a in world.get_actors().filter("vehicle.*"):
        if a.attributes.get("role_name") in ("ego_vehicle", "hero"):
            return a
    return None


def ego_spawn_location() -> "carla.Location":
    """Ego spawn point from config/objects.json (ROS frame → CARLA: y negated)."""
    spec = json.loads(OBJECTS_JSON.read_text())
    for obj in spec["objects"]:
        if obj.get("id") == "ego_vehicle" and "spawn_point" in obj:
            sp = obj["spawn_point"]
            loc = carla.Location(x=sp["x"], y=-sp["y"], z=sp.get("z", 0.5))
            return loc, -sp.get("yaw", 0.0)          # CARLA yaw (degrees)
    raise RuntimeError(f"no ego_vehicle spawn_point in {OBJECTS_JSON}")


def match_heading(wp, carla_yaw_deg: float):
    """Same rule as planner_node._correct_lane_direction: if the lane points
    more than 90° away from the vehicle heading, use the left lane."""
    diff = (wp.transform.rotation.yaw - carla_yaw_deg + 180) % 360 - 180
    if abs(diff) > 90:
        other = wp.get_left_lane()
        if other is not None and other.lane_type == carla.LaneType.Driving:
            return other
    return wp


def clear(world) -> int:
    n = 0
    for a in world.get_actors().filter("vehicle.*"):
        if a.attributes.get("role_name") == ROLE:
            a.destroy()
            n += 1
    return n


def curvature_hint(start_wp, end_wp) -> str:
    dyaw = (end_wp.transform.rotation.yaw - start_wp.transform.rotation.yaw + 180) % 360 - 180
    return "straight" if abs(dyaw) < 10 else f"curving {dyaw:+.0f}° (CARLA yaw)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("scenario", nargs="?", choices=[*SCENARIOS, "custom"])
    ap.add_argument("--distance", type=float, help="metres ahead of the ego along its lane")
    ap.add_argument("--lateral", type=float, help="metres LEFT of lane centre (custom/override)")
    ap.add_argument("--blueprint", default="vehicle.audi.a2")
    ap.add_argument("--host", default="172.25.80.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--keep", action="store_true", help="leave the cars when this script exits")
    ap.add_argument("--clear", action="store_true", help="destroy previously spawned scenario cars")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    removed = clear(world)
    if removed:
        print(f"Removed {removed} previous scenario car(s).")
    if args.clear:
        return
    if args.scenario is None:
        ap.error("give a scenario (blocked | partial | custom) or --clear")

    ego = find_ego(world)
    if ego is not None:
        ref_loc, ref_yaw, ref_name = ego.get_location(), ego.get_transform().rotation.yaw, "live ego"
    else:
        (ref_loc, ref_yaw), ref_name = ego_spawn_location(), "ego spawn point (config/objects.json)"
    print(f"Placing relative to the {ref_name}.")

    placements = list(SCENARIOS.get(args.scenario, [(30.0, 0.0)]))
    if args.distance is not None or args.lateral is not None:
        d0, l0 = placements[0]
        placements[0] = (args.distance if args.distance is not None else d0,
                         args.lateral if args.lateral is not None else l0)

    cmap = world.get_map()
    ego_wp = match_heading(
        cmap.get_waypoint(ref_loc, project_to_road=True, lane_type=carla.LaneType.Driving),
        ref_yaw)
    bp = world.get_blueprint_library().find(args.blueprint)
    bp.set_attribute("role_name", ROLE)

    spawned = []
    for dist, lat in placements:
        wp, travelled, junction = walk_route(ego_wp, dist)
        if wp is None:
            print(f"  road ends after {travelled:.0f} m — cannot place a car at {dist:.0f} m")
            continue
        loc = lateral_location(wp, lat)
        loc.z += 0.3
        tf = carla.Transform(loc, wp.transform.rotation)
        actor = world.try_spawn_actor(bp, tf)
        if actor is None:
            print(f"  spawn failed at {dist:.0f} m (occupied?) — try another --distance")
            continue
        actor.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
        spawned.append(actor)
        # ROS frame for comparison with /perception/obstacles: y negated
        print(f"  {args.blueprint} at {dist:.0f} m ahead, {lat:+.1f} m left of centre "
              f"(road {curvature_hint(ego_wp, wp)}{', JUNCTION on the way' if junction else ''})\n"
              f"    ROS map frame: x={loc.x:.1f} y={-loc.y:.1f}   lane width {wp.lane_width:.1f} m")

    if not spawned:
        sys.exit(1)
    if args.keep:
        print("Kept. Remove later with:  python3 scripts/spawn_scenario.py --clear")
        return

    print("Scenario active — Ctrl+C to remove the car(s).")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        for a in spawned:
            a.destroy()
        print("Scenario car(s) removed.")


if __name__ == "__main__":
    main()
