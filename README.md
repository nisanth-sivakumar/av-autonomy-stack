# AV Autonomy Stack

[![CI](https://github.com/nisanth-sivakumar/av-autonomy-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/nisanth-sivakumar/av-autonomy-stack/actions/workflows/ci.yml)
[![Offline scenarios](https://github.com/nisanth-sivakumar/av-autonomy-stack/actions/workflows/nightly.yml/badge.svg)](https://github.com/nisanth-sivakumar/av-autonomy-stack/actions/workflows/nightly.yml)

A perception → planning → control stack that drives a Tesla Model 3 in CARLA 0.9.15 through
ROS 2 Humble: three interchangeable planners (road graph, lattice, road-constrained MPPI), a
behavior layer that follows and overtakes by changing lanes, and two controllers (Python Pure
Pursuit and a C++ LTV MPC on OSQP). Every change is gated by unit tests and 30 closed-loop
scenarios that run the real nodes without CARLA, and validated in CARLA against written pass
criteria.

![Demo: MPPI passes a partially blocking car, then follows a lead car](media/demo.gif)

## Results

**In CARLA** (Town10HD_Opt, each run against written pass criteria):

| | Before (May 2026 bags) | Now |
|---|---|---|
| MPPI collisions | 7 of 7 runs | **0** while driving, in every validation run since 2026-11-05 |
| Junction corner-cutting, max | 0.77 m | **0.36–0.40 m** (Pure Pursuit), **0.26–0.30 m** (MPC) |
| Lane-offset estimate, mean \|error\| | 0.68 m | **0.22 m** |

| CARLA run (2026-11-07, Pure Pursuit) | road_graph | lattice | mppi |
|---|---|---|---|
| Empty road (~480 m with junction turns): CTE mean / max | 0.07 / 0.36 m | 0.07 / 0.36 m | 0.07 / 0.40 m |
| Car parked in the lane centre | stops, gap 2.9 m | stops, 3.4 m | stops, 3.2 m |
| Car 1.0 m into the lane | stops, 3.2 m | **passes**, 0.68 m body clearance | **passes**, 0.40 m |
| Lead car at 1 m/s, then parks | follows, stops ~4 m behind | same | same |
| Goal ~370 m away by A* | 1.30 m short | 1.28 m short | 1.27 m short |

Overtaking (2026-11-08): lattice and MPPI pass a slow lead by changing lanes, stay behind when
the other lane is occupied, abort when a car closes in from behind, and never change into an
oncoming lane (8/8 runs, 0 collisions). The goal stop now ends within 0.1 m offline (4.0.5;
not yet re-measured in CARLA).

**MPC vs Pure Pursuit** (same plans, CARLA's vehicle geometry):

| | Pure Pursuit | C++ LTV MPC |
|---|---|---|
| CARLA junction turns, max tracking error | 0.36–0.40 m | **0.26–0.30 m** |
| CARLA steering rate (road_graph, lattice) | 0.11 rad/s | **0.04 rad/s** |
| Offline scenarios passed | 26/30 | **30/30** |
| Offline max / mean CTE vs plan | 0.60 / 0.127 m | **0.30 / 0.030 m** |
| Randomized sweep (300 runs) | 294/300, 0 collisions | 294/300, 0 collisions |
| Compute p99 (50 ms control period) | 0.09 ms | 0.38 ms |

## Architecture

```mermaid
flowchart LR
    CARLA[CARLA 0.9.15<br/>Windows host] <-->|carla-ros-bridge| B((ROS 2))
    B -->|odometry, segmentation image, /clock| P[perception_node<br/>objects + velocity,<br/>lane offset]
    P -->|/perception/objects| PL[planner_node<br/>A* route · behavior layer ·<br/>road_graph / lattice / MPPI]
    B -->|odometry| PL
    PL -->|/planning/trajectory| C[controller<br/>Pure Pursuit (Python)<br/>or LTV MPC (C++)]
    P -->|/perception/objects| C
    C -->|vehicle_control_cmd| B
```

Each node is a thin ROS wrapper around a ROS-free module, and the unit tests target those
modules. All nodes run on simulated time.

## Bugs found by validation

- **MPPI's obstacle cost was discontinuous** at the clearance radius and only looked at the
  closest waypoint: the optimum sat on the boundary and the swerve came too late for the car to
  follow (an offline collision). One of several MPPI bugs behind the early CARLA collisions.
- **A bridge exception froze the whole world.** The DVS camera used `numpy.bool` (removed in
  NumPy 1.24); the exception killed the bridge's synchronous tick thread, and nothing else ticks
  CARLA in synchronous mode.
- **The lattice planner walked the car onto the kerb** in junction turns: each plan restarted
  from the car's tracking error, so cutting the corner compounded. Fixed with plan continuity.
- **The CTE metric overstated corner-cutting by ~0.7 m** by extrapolating the plan's first
  segment backwards; the "1.4 m" junction error was really 0.77 m.
- **The MPC's reference was wrong at every curve entry and exit** (23° heading error on a car
  that was on the path): it extrapolated the road behind from the road ahead. The offline
  scenarios passed regardless; only CARLA showed it.
- **Stopping a run let the car drive on** into a wall: CARLA keeps applying the last command,
  the stop brake was published before discovery matched and never arrived, and scripted runs
  inherited SIGINT *ignored*. Both controllers now brake on shutdown.

## Setup

Needs Ubuntu 22.04 (or WSL2) with [ROS 2 Humble](https://docs.ros.org/en/humble/Installation.html).
CARLA is only needed to drive in the simulator, not to build, test or run the offline scenarios.

```bash
git clone https://github.com/nisanth-sivakumar/av-autonomy-stack.git ~/av-autonomy-stack
cd ~/av-autonomy-stack
./scripts/setup.sh              # ros-bridge submodule + local patch + rosdep (add --analysis for analyze_run.py)
./scripts/check.sh              # build + unit tests + 30 offline scenarios, both controllers (~9 min; --quick: build + tests)
```

`patches/ros-bridge-local.patch` relaxes the bridge's exact-CARLA-version check (client and
server are 0.9.15, the bridge commit targets 0.9.13), fixes the `pcl_recorder` build and the
DVS camera's `numpy.bool`, and keeps the synchronous tick loop alive on exceptions.

## Running in CARLA

Install the CARLA 0.9.15 Python client (`pip3 install --user carla==0.9.15`) and start
`CarlaUE4.exe` (0.9.15). The CARLA host address (`172.25.80.1`, the Windows host seen from
WSL2) is set in the perception and planner launch files.

```bash
source /opt/ros/humble/setup.bash && source carla_ros_ws/install/setup.bash
# terminal 1: the bridge
ros2 launch carla_ros_bridge carla_ros_bridge.launch.py host:=172.25.80.1 port:=2000 town:=Town10HD_Opt
# terminal 2 (any directory): spawn, perception, planner, controller, bag recording
./scripts/reset_run.sh mppi                       # road_graph | lattice | mppi; Ctrl+C to stop
./scripts/reset_run.sh mppi 30 -28.2              # ...and drive to a goal (x, y, ROS map frame)
CONTROLLER=mpc ./scripts/reset_run.sh lattice     # the C++ MPC instead of Pure Pursuit
python3 scripts/spawn_scenario.py partial --keep  # parked-car scenarios (spawn before the run)
python3 scripts/analyze_run.py bags/<run_dir>     # CTE, collisions, lane invasions, plots
```

## Testing without CARLA

```bash
./scripts/check.sh                                       # the gate run before every commit
python3 scripts/offline_sim/run_scenarios.py --rate 4    # the 30 closed-loop scenarios alone
python3 scripts/offline_sim/run_scenarios.py --sweep 100 --seed 2026 --rate 4 --jobs 3 --workdir /tmp/sw
```

The offline simulation runs the real nodes and launch files against a stub `carla` module
(a synthetic road with one or two lanes, parked and moving cars) and a kinematic vehicle with a
rendered segmentation camera. The scenarios cover curves, a tight junction-like turn, blocked and
partially blocked lanes, lead-car following, overtaking (free, blocked and aborted lane changes,
never into oncoming traffic) and stopping at a goal, for all three planners. The randomized
sweep draws variants of them and checks invariants (no collision, clearances, no oncoming lane
change).

| Scenario | road_graph | lattice | mppi |
|---|---|---|---|
| curve (R 30 m) / tight turn (R 10 m): max CTE | 0.16 / 0.36 m | 0.16 / 0.36 m | 0.17 / 0.37 m |
| car parked in the lane | stops, gap 3.0 m | stops, 3.4 m | stops, 3.1 m |
| car 1.0 m into the lane, on the curve | stops | **passes** | **passes** |
| lead at 1.5 m/s | follows at the 1 s gap | same | same |
| slow lead, free left lane | follows | **changes lanes, passes** | **changes lanes, passes** |
| ... other lane occupied / car closing in from behind | follows | no change / **aborts** | no change / **aborts** |
| slow lead, neighbour lane oncoming | follows | no change | no change |
| goal on the curve: stop error | +0.0 m | −0.1 m | +0.0 m |

## Key design decisions

- **Road-constrained MPPI.** Sampling steering sequences through a bicycle model failed on
  turns: most samples go straight, so the optimiser converged to "mostly straight". Optimising
  lateral offsets along the road-graph centre line keeps every sample on the road by construction.
- **Simulated perception with controlled noise.** Objects come from CARLA's ground truth plus
  Gaussian noise on position and velocity, so the planners and controllers are tested against
  noisy inputs without a detector in the loop. Lane offset is real vision on the segmentation
  camera.
- **Planner/controller separation.** Controllers follow `/planning/trajectory` and know nothing
  about the planner; the obstacle stop in the controller is the safety net if a plan is wrong.
- **Margins ordered on purpose:** ego half-width 1.08 m < controller corridor 1.35 m < lattice
  1.75 m < MPPI 1.95 m. If a planner's margin drops to the corridor, perception noise makes the
  controller brake beside obstacles the planner routes around.
- **Simulated time everywhere.** CARLA in synchronous mode runs about 2× real time; wall-clock
  timers made the planner and perception run at half rate.

