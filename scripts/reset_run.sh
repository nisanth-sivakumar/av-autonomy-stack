#!/usr/bin/env bash
# scripts/reset_run.sh
# ---------------------
# Full pipeline launcher: bag → spawn → perception → planner → controller
#
# Usage:
#   ./scripts/reset_run.sh                  # road_graph planner (default)
#   ./scripts/reset_run.sh lattice          # lattice planner
#   ./scripts/reset_run.sh mppi             # road-constrained MPPI
#
# Run from: av-autonomy-stack/

set -e

PLANNER_TYPE="${1:-road_graph}"

source carla_ros_ws/install/setup.bash

mkdir -p bags

# ---- Stop everything from the previous run -----------------------------
echo "==> Stopping previous run..."
pkill -f "perception_node"          2>/dev/null || true
pkill -f "road_graph_planner"       2>/dev/null || true
pkill -f "pure_pursuit_controller"  2>/dev/null || true
pkill -f "carla_spawn_objects"      2>/dev/null || true
pkill -f "ros2 bag record"          2>/dev/null || true

echo "==> Waiting for CARLA to clean up actors (3 s)..."
sleep 3

# ---- Start bag recording -----------------------------------------------
RUN_ID="run_${PLANNER_TYPE}_$(date +%Y%m%d_%H%M%S)"
BAG_PATH="bags/$RUN_ID"

echo "==> Starting bag recording → $BAG_PATH"
ros2 bag record \
    --output "$BAG_PATH" \
    /carla/ego_vehicle/odometry \
    /carla/ego_vehicle/vehicle_control_cmd \
    /carla/ego_vehicle/speedometer \
    /carla/ego_vehicle/collision \
    /carla/ego_vehicle/lane_invasion \
    /planning/trajectory \
    /perception/obstacles \
    /perception/lane_offset &
BAG_PID=$!

sleep 1

# Apply custom objects.json
cp ~/av-autonomy-stack/config/objects.json \
   ~/av-autonomy-stack/carla_ros_ws/src/ros-bridge/carla_spawn_objects/config/objects.json

# ---- Relaunch spawn objects --------------------------------------------
echo "==> Relaunching spawn_objects..."
ros2 launch carla_spawn_objects carla_spawn_objects.launch.py &
SPAWN_PID=$!

echo "==> Waiting for objects to spawn (5 s)..."
sleep 5

# ---- Relaunch perception -----------------------------------------------
echo "==> Relaunching perception..."
ros2 launch av_perception perception.launch.py &
PERCEP_PID=$!

echo "==> Waiting for perception to connect to CARLA (3 s)..."
sleep 3

# ---- Relaunch planner --------------------------------------------------
echo "==> Relaunching planner (type: $PLANNER_TYPE)..."
ros2 launch av_planning planner.launch.py planner_type:=$PLANNER_TYPE &
PLANNER_PID=$!

echo "==> Waiting for planner to connect to CARLA and generate first trajectory (4 s)..."
sleep 4

# ---- Relaunch controller -----------------------------------------------
echo "==> Relaunching controller..."
ros2 launch av_control controller.launch.py &
CTRL_PID=$!

# ---- Summary -----------------------------------------------------------
echo ""
echo "==> Run started."
echo "    Planner     : $PLANNER_TYPE"
echo "    Bag         : $BAG_PATH"
echo "    spawn PID   : $SPAWN_PID"
echo "    perception  : $PERCEP_PID"
echo "    planner PID : $PLANNER_PID"
echo "    controller  : $CTRL_PID"
echo "    bag PID     : $BAG_PID"
echo ""
echo "    Press Ctrl+C to stop this run and save the bag."

trap "echo '';
      echo '==> Stopping run...';
      ros2 topic pub --once /carla/ego_vehicle/vehicle_control_cmd \
          carla_msgs/msg/CarlaEgoVehicleControl \
          '{throttle: 0.0, steer: 0.0, brake: 1.0}' 2>/dev/null;
      sleep 0.5;
      kill $CTRL_PID $PLANNER_PID $PERCEP_PID $SPAWN_PID $BAG_PID 2>/dev/null;
      echo '==> Bag saved to $BAG_PATH';
      echo '==> Run stopped.'" INT

wait