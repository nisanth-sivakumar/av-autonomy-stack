# Autonomous Driving Stack

A closed-loop autonomous driving stack built from scratch in Python and ROS2, running inside the CARLA simulator. The stack implements a full sensor → perception → planning → control pipeline, with three interchangeable planners that can be benchmarked against each other at runtime.

---

## Architecture

```
CARLA Simulator (Windows)
        │
        │  TCP  172.25.80.1:2000
        │
carla_ros_bridge  (WSL / ROS2 Humble)
        │
        ├──▶  /perception/obstacles  ──────▶  ┌─────────────────┐
        │                                     │  perception_node│
        ├──▶  /carla/.../semantic_seg ──────▶ │  (Phase 3)      │
        │                                     └────────┬────────┘
        │                                              │ /perception/obstacles
        │                                              │ /perception/lane_offset
        │                                              ▼
        │                                     ┌─────────────────┐
        │                                     │  planner_node   │
        │                                     │  road_graph /   │
        │                                     │  lattice / mppi │
        │                                     └────────┬────────┘
        │                                              │ /planning/trajectory
        │                                              ▼
        │                                     ┌─────────────────┐
        └──▶  /carla/.../odometry  ──────────▶│ controller_node │
                                              │  Pure Pursuit   │
                                              └────────┬────────┘
                                                       │ /carla/.../vehicle_control_cmd
                                                       ▼
                                                  CARLA Vehicle
```

Each layer communicates exclusively through ROS2 topics. Planners are swappable at launch time with a single argument.

---

## Stack Components

### Control — `av_control`
Pure Pursuit lateral controller with a proportional speed controller.

- Subscribes to `nav_msgs/Path` on `/planning/trajectory`
- Quaternion → yaw conversion, body-frame transform, forward waypoint filtering
- Stuck detection with braking fallback
- Trajectory timeout guard — brakes if the planner goes silent

### Planning — `av_planning`
Three interchangeable planners, selectable via `planner_type` launch argument.

**Road Graph** (`road_graph`) — Phase 2a baseline  
Walks the CARLA road graph using `get_waypoint()` + `.next()`, respecting lane direction and road topology. Used as the centre-line reference by all three planners.

**Lattice Planner** (`lattice`) — Phase 2b  
Samples `N` candidate trajectories as lateral offsets from the road-graph centre-line, scores each on lane deviation, curvature, and smoothness using a weighted cost function, and returns the minimum-cost feasible path. Obstacle avoidance is built into the feasibility check — any path passing within `min_clearance_m` of a detected obstacle is marked infeasible.

**Road-Constrained MPPI** (`mppi`) — Phase 2b  
Model Predictive Path Integral optimisation over lateral offsets along the road-graph centre-line. Samples `K=400` perturbation sequences, builds candidate trajectories by laterally shifting the centre-line, scores each with a cost function (lane deviation, smoothness, deviation penalty, obstacle clearance), and updates the nominal offset sequence via importance weighting. Warm-starts between planning cycles for fast convergence at 10 Hz.

Parameterising over road-graph offsets rather than free-space control inputs guarantees road-following by construction — all candidate trajectories follow the road regardless of whether the optimiser has converged.

### Perception — `av_perception`
**Object detection**  
Queries the CARLA Python API for nearby vehicles and pedestrians within a configurable detection range, adds Gaussian position noise to simulate real sensor uncertainty, and publishes as `geometry_msgs/PoseArray` on `/perception/obstacles`. Noise standard deviation is tunable to model different sensor types (e.g. 0.2 m for LiDAR, 0.5 m for camera).

**Lane detection**  
Processes the semantic segmentation camera image from CARLA. Extracts the road-line semantic label (class 6) from the R channel of the image, isolates lane marking pixels in a configurable bottom ROI, and computes the signed lateral offset of the ego vehicle from the lane centre. Published as `std_msgs/Float32` on `/perception/lane_offset`. Includes a colour-matching fallback for cross-version robustness and publishes an annotated debug image on `/perception/debug/image`.

---

## Repository Structure

```
av-autonomy-stack/
├── config/
│   └── objects.json              # Ego vehicle + sensor spawn config
├── scripts/
│   ├── reset_run.sh              # Full pipeline launcher (see Usage)
│   └── analyze_run.py            # Metrics + plots from bag files
├── bags/                         # Auto-timestamped run recordings
└── carla_ros_ws/
    └── src/
        ├── ros-bridge/           # carla_ros_bridge (git submodule)
        ├── av_control/           # Pure Pursuit controller
        │   ├── av_control/
        │   │   └── controller_node.py
        │   └── launch/
        │       └── controller.launch.py
        ├── av_planning/          # Three interchangeable planners
        │   ├── av_planning/
        │   │   ├── planner_node.py
        │   │   ├── lattice_planner.py
        │   │   └── mppi_planner.py
        │   └── launch/
        │       └── planner.launch.py
        └── av_perception/        # Perception layer
            ├── av_perception/
            │   └── perception_node.py
            └── launch/
                └── perception.launch.py
```

---

## Environment

| Component | Version |
|---|---|
| Simulator | CARLA 0.9.15 (Windows) |
| ROS | ROS2 Humble (WSL / Ubuntu 22.04) |
| Bridge | carla_ros_bridge (git submodule) |
| Map | Town10HD_Opt |
| Ego vehicle | Tesla Model 3 (wheelbase = 2.875 m) |
| Python | 3.10 |

CARLA runs on Windows. ROS2 and all stack nodes run in WSL2. Communication uses the Windows host bridge IP (`172.25.80.1:2000`).

---

## Setup

### Prerequisites

```bash
# ROS2 Humble — follow https://docs.ros.org/en/humble/Installation.html

# System dependencies
sudo apt install ros-humble-cv-bridge python3-opencv python3-numpy

# Clone the repo (with submodule)
git clone --recurse-submodules https://github.com/your-username/av-autonomy-stack.git
cd av-autonomy-stack
```

### Build

```bash
cd carla_ros_ws
colcon build --packages-select av_perception av_planning av_control
source install/setup.bash
```

### CARLA Python API

Add the CARLA Python egg to your `~/.bashrc`:

```bash
export PYTHONPATH=$PYTHONPATH:~/CARLA_0.9.15/PythonAPI/carla/dist/<egg_name>.egg
```

Find the egg name with:
```bash
ls ~/CARLA_0.9.15/PythonAPI/carla/dist/
```

---

## Usage

Start CARLA on Windows, then launch the full pipeline from WSL:

```bash
# Start the ROS-CARLA bridge (separate terminal)
source carla_ros_ws/install/setup.bash
ros2 launch carla_ros_bridge carla_ros_bridge.launch.py \
    host:=172.25.80.1 port:=2000 passive:=True

# Launch the full pipeline (separate terminal)
./scripts/reset_run.sh                  # road_graph planner (default)
./scripts/reset_run.sh lattice          # lattice planner
./scripts/reset_run.sh mppi             # road-constrained MPPI
```

`reset_run.sh` handles everything in sequence: bag recording → spawn → perception → planner → controller. Press `Ctrl+C` to stop the run cleanly — a brake command is sent before all nodes are killed, and the bag is saved automatically.

### Switching planners mid-project

All three planners share the same `/planning/trajectory` interface. The controller has no knowledge of which planner is active:

```bash
# Benchmark all three back-to-back
./scripts/reset_run.sh road_graph
./scripts/reset_run.sh lattice
./scripts/reset_run.sh mppi
```

### Analysing a run

```bash
python3 scripts/analyze_run.py bags/run_mppi_20260507_174908
```

Outputs: cross-track error (mean/max), speed statistics, steering jerk, distance travelled, collision and lane invasion counts, and a `run_plots.png`.

---

## Perception Topics

| Topic | Type | Description |
|---|---|---|
| `/perception/obstacles` | `geometry_msgs/PoseArray` | Detected obstacle positions in map frame |
| `/perception/lane_offset` | `std_msgs/Float32` | Signed lateral offset from lane centre (m) |
| `/perception/debug/image` | `sensor_msgs/Image` | Annotated semantic seg image for RViz |

---

## Metrics Collected per Run

Every run is recorded as a ROS2 bag. `analyze_run.py` extracts:

- **Cross-track error** — deviation from the road centre-line
- **Speed** — mean, max, and distribution
- **Steering jerk** — rate of change of steering command
- **Distance travelled**
- **Collision count** — from CARLA's collision sensor
- **Lane invasion count** — from CARLA's lane invasion sensor
- **Obstacle avoidance margin** — minimum distance to detected obstacles (Phase 3)
- **Lane offset** — from perception, logged independently of road graph

These metrics make it straightforward to compare the three planners quantitatively across identical runs.

---

## Key Design Decisions

**Planner/controller separation.** The controller is a pure waypoint follower with no knowledge of the planning algorithm. Planners are swapped by changing one launch argument — no controller changes required.

**Road-constrained MPPI.** The conventional kinematic-rollout MPPI (sampling control sequences through a bicycle model) failed on turns in this setup because the bicycle model rollout is unconstrained — most samples go straight, so the optimiser converges to "mostly straight" even when the road turns. Parameterising over lateral offsets from the road-graph centre-line instead guarantees road-following by construction. This is more honest to how production AV stacks use MPPI, where the trajectory search space is constrained to driveable regions.

**Perception with controlled noise.** Rather than using CARLA's ground-truth object positions directly, the perception node adds Gaussian noise before publishing obstacles to the planner. This makes the obstacle avoidance behaviour more realistic and allows testing planner robustness to noisy inputs by varying `position_noise_std_m`.

**QoS matching.** The carla_ros_bridge publishes odometry with `RELIABLE` + `VOLATILE` QoS. All subscribers in this stack explicitly declare matching QoS profiles — passing a bare integer depth to `create_subscription` causes DDS to silently drop the connection in ROS2 Humble.

---
