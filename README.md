# AV Autonomy Stack

A modular perception → planning → control stack for an ego vehicle in CARLA 0.9.15,
built as ROS 2 Humble Python packages on top of `carla-ros-bridge`.

- **Perception** (`av_perception`): simulated object detection (CARLA actors as boxes with
  velocity, plus noise) and vision-based lane offset from the semantic segmentation camera.
- **Planning** (`av_planning`): road-graph centre line from the CARLA map with an optional
  A* route to a goal, plus lattice and road-constrained MPPI planners that steer around
  (predicted) obstacles.
- **Control** (`av_control`): Pure Pursuit with a speed- and curvature-adaptive lookahead,
  and speed control that stops for obstacles, follows lead cars and stops at the goal.

See [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for the architecture, what's
implemented and known limitations.

## Setup

```bash
git clone <this repo> ~/av-autonomy-stack && cd ~/av-autonomy-stack
git submodule update --init --recursive
git -C carla_ros_ws/src/ros-bridge apply ../../../patches/ros-bridge-local.patch   # local bridge fixes

source /opt/ros/humble/setup.bash
cd carla_ros_ws && colcon build --symlink-install && source install/setup.bash
pip3 install --user rosbags pandas "numpy<2"
```

`patches/ros-bridge-local.patch` relaxes the bridge's exact-CARLA-version check (client
and server are 0.9.15, the bridge commit targets 0.9.13), defaults to asynchronous mode
with a 20 s timeout, and fixes a `pcl_recorder` build error.

## Running in CARLA

```bash
# Windows: start CarlaUE4.exe (0.9.15), map Town10HD_Opt
# WSL terminal 1:
ros2 launch carla_ros_bridge carla_ros_bridge.launch.py host:=172.25.80.1 port:=2000 town:=Town10HD_Opt
# WSL terminal 2 (any directory):
./scripts/reset_run.sh mppi          # road_graph | lattice | mppi; Ctrl+C to stop
./scripts/reset_run.sh mppi 30 -28.2 # ...and drive to a goal (x, y in the ROS map frame)
python3 scripts/analyze_run.py bags/<run_dir>
```

## Testing without CARLA

```bash
source carla_ros_ws/install/setup.bash
cd carla_ros_ws && colcon test --packages-select av_control av_planning av_perception \
    && colcon test-result --verbose                       # unit tests
cd .. && python3 scripts/offline_sim/run_scenarios.py     # closed-loop scenarios (~11 min)
```

The offline simulation runs the real nodes and launch files against a fake `carla`
module (synthetic road and parked cars) and a kinematic vehicle with a rendered
segmentation camera. It checks lane keeping on a gentle and a tight curve, perception
accuracy, stopping for a blocked lane, passing a partially blocking car, following a
moving lead car and stopping at a goal.

CARLA validation steps and pass criteria: [docs/CARLA_VALIDATION.md](docs/CARLA_VALIDATION.md).
