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

Moving lead car (CARLA version of the offline "lead" scenario):
    python3 scripts/spawn_scenario.py custom --distance 45 --lateral 0 --speed 1.5 --travel 25
    # then ./scripts/reset_run.sh <planner> in another terminal
The car waits (braked) until the ego starts moving, then drives straight
ahead at constant velocity for --travel metres and stops.  It does not
steer, so place it on a straight with that much road ahead.

Multi-car / neighbour-lane scenarios (Phase 4.1 overtaking):
    --lane left|right   place the car in the neighbour lane (CARLA's lane graph)
    --distance -15      negative = behind the reference, in that lane
    --add               don't remove existing scenario cars first (one
                        invocation per car; each moving car keeps its own process)
    --trigger-speed V   a moving car switches to V m/s when the ego first leaves
                        its lane (> 0.5 m from its centre) — closes the gap
                        during a lane change (abort scenario)

Lateral convention: + = LEFT of the lane centre (same as the planners).
"""

import argparse
import json
import os
import math
import signal
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
    """Follow the lane `distance_m` ahead like the planner does (first branch);
    a negative distance walks backwards along the lane."""
    wp, travelled, junction = start_wp, 0.0, False
    backwards = distance_m < 0
    distance_m = abs(distance_m)
    while travelled < distance_m:
        step = min(step_m, distance_m - travelled)
        nxt = wp.previous(step) if backwards else wp.next(step)
        if not nxt:
            return None, travelled, junction
        wp = nxt[0]
        travelled += step_m
        junction = junction or getattr(wp, "is_junction", False)
    return wp, travelled, junction


def find_ego(world):
    for a in fresh_actors(world).filter("vehicle.*"):
        if a.attributes.get("role_name") in ("ego_vehicle", "hero"):
            return a
    return None


def ego_spawn_location() -> "carla.Location":
    """Ego spawn point: $EGO_SPAWN ("x,y,z,roll,pitch,yaw", as reset_run.sh takes it)
    or config/objects.json, both in the ROS frame (→ CARLA: y and yaw negated)."""
    override = os.environ.get("EGO_SPAWN")
    if override:
        x, y, z, _, _, yaw = (float(v) for v in override.split(","))
        return carla.Location(x=x, y=-y, z=z), -yaw
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


def fresh_actors(world):
    """
    Actor list after the next tick.  A new client's first get_actors() can be
    empty or stale in synchronous mode until it has seen a tick — `--clear`
    silently missed cars because of this.
    """
    try:
        world.wait_for_tick(5.0)
    except RuntimeError:
        pass   # no tick within 5 s (bridge stalled?) — use what we have
    return world.get_actors()


def remove(actor):
    try:
        actor.disable_constant_velocity()    # persists after this script exits otherwise
    except RuntimeError:
        pass
    actor.destroy()


def clear(world) -> int:
    n = 0
    for a in fresh_actors(world).filter("vehicle.*"):
        if a.attributes.get("role_name") == ROLE:
            remove(a)
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
    ap.add_argument("--speed", type=float, default=0.0,
                    help="make the car a lead vehicle: drive at this speed (m/s) once the ego moves")
    ap.add_argument("--travel", type=float, default=25.0,
                    help="lead vehicle: metres to drive before stopping (default 25)")
    ap.add_argument("--start-now", action="store_true",
                    help="lead vehicle: start driving immediately (testing) instead of waiting for the ego")
    ap.add_argument("--clear", action="store_true", help="destroy previously spawned scenario cars")
    ap.add_argument("--add", action="store_true",
                    help="keep existing scenario cars (default: remove them first)")
    ap.add_argument("--lane", choices=("left", "right"),
                    help="put the car in this neighbour lane of the reference lane")
    ap.add_argument("--trigger-speed", type=float, default=None,
                    help="moving car: switch to this speed when the ego first leaves its lane")
    args = ap.parse_args()
    # SIGTERM (e.g. from a script) must clean up like Ctrl+C does
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    removed = 0 if args.add and not args.clear else clear(world)
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
        (ref_loc, ref_yaw), ref_name = ego_spawn_location(), (
            "ego spawn point ($EGO_SPAWN)" if os.environ.get("EGO_SPAWN")
            else "ego spawn point (config/objects.json)")
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
        road_wp = wp                       # the reference lane, for the road hint
        if args.lane:
            nb = wp.get_left_lane() if args.lane == "left" else wp.get_right_lane()
            if nb is None or nb.lane_type != carla.LaneType.Driving:
                print(f"  no {args.lane} driving lane at {dist:.0f} m")
                continue
            same = (nb.lane_id > 0) == (wp.lane_id > 0)
            print(f"  {args.lane} neighbour lane {nb.lane_id} "
                  f"({'same direction' if same else 'ONCOMING'})")
            wp = nb
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
              f"(road {curvature_hint(ego_wp, road_wp)}{', JUNCTION on the way' if junction else ''})\n"
              f"    ROS map frame: x={loc.x:.1f} y={-loc.y:.1f}   lane width {wp.lane_width:.1f} m")

    if not spawned:
        sys.exit(1)
    if args.speed > 0.0:
        if args.keep:
            print("--speed needs this script to keep running; ignoring --keep.")
        drive_lead(world, spawned[0], args.speed, args.travel, args.start_now,
                   trigger_speed=args.trigger_speed)
        return
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
            remove(a)
        print("Scenario car(s) removed.")


def ego_left_lane(cmap, ego, lane_id) -> bool:
    """Has the ego left lane `lane_id` (> 0.5 m from its centre, or another lane)?"""
    loc = ego.get_location()
    wp = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    return wp.lane_id != lane_id or loc.distance(wp.transform.location) > 0.5


def drive_lead(world, car, speed, travel, start_now=False, trigger_speed=None):
    """Hold the car until the ego moves, then drive it ahead at `speed` for `travel` m
    (switching to `trigger_speed` when the ego first leaves its lane)."""
    print(f"Lead vehicle armed: waits for the ego to move, then {speed:.1f} m/s for {travel:.0f} m. "
          "Ctrl+C to remove it.", flush=True)
    try:
        while not start_now:
            ego = find_ego(world)
            if ego is not None:
                v = ego.get_velocity()
                if (v.x ** 2 + v.y ** 2) ** 0.5 > 0.5:
                    break
            time.sleep(0.2)
        # A just-spawned actor reports (0, 0, 0) until the world has ticked;
        # taking that as the start made the travel check pass instantly.
        world.wait_for_tick(5.0)
        start = car.get_location()
        car.apply_control(carla.VehicleControl(hand_brake=False, brake=0.0))
        car.enable_constant_velocity(carla.Vector3D(speed, 0.0, 0.0))   # car's local x = forward
        print("Lead vehicle driving.", flush=True)
        cmap, ego = world.get_map(), find_ego(world)
        ego_lane = None if ego is None else cmap.get_waypoint(ego.get_location()).lane_id
        triggered = trigger_speed is None or ego is None
        while car.get_location().distance(start) < travel:
            if not triggered and ego_left_lane(cmap, ego, ego_lane):
                car.enable_constant_velocity(carla.Vector3D(trigger_speed, 0.0, 0.0))
                print(f"Ego left its lane — now {trigger_speed:.1f} m/s.", flush=True)
                triggered = True
            time.sleep(0.1)
        car.disable_constant_velocity()
        car.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
        print(f"Lead vehicle stopped after {travel:.0f} m (parked; Ctrl+C to remove).", flush=True)
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        remove(car)
        print("Lead vehicle removed.", flush=True)


if __name__ == "__main__":
    main()
