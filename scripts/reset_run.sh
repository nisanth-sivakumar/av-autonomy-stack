#!/usr/bin/env bash
# scripts/reset_run.sh
# ---------------------
# Kills the spawn_objects node, controller, and any active bag recording,
# waits for CARLA to despawn all actors, then relaunches everything.
# A new timestamped bag is recorded for every run automatically.
#
# Usage  : ./scripts/reset_run.sh
# Run from: av-autonomy-stack/

set -e

source carla_ros_ws/install/setup.bash

# ---- Create bags directory if it doesn't exist -------------------------
mkdir -p bags

# ---- Stop everything from the previous run -----------------------------
echo "==> Stopping previous run..."
pkill -f "pure_pursuit_controller"  2>/dev/null || true
pkill -f "carla_spawn_objects"      2>/dev/null || true
pkill -f "ros2 bag record"          2>/dev/null || true

echo "==> Waiting for CARLA to clean up actors (3 s)..."
sleep 3

# ---- Start bag recording -----------------------------------------------
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
BAG_PATH="bags/$RUN_ID"

echo "==> Starting bag recording → $BAG_PATH"
ros2 bag record \
    --output "$BAG_PATH" \
    /carla/ego_vehicle/odometry \
    /carla/ego_vehicle/vehicle_control_cmd \
    /carla/ego_vehicle/speedometer \
    /carla/ego_vehicle/collision \
    /carla/ego_vehicle/lane_invasion \
    &
BAG_PID=$!

# Give the recorder a moment to initialise before the car starts moving
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

# ---- Relaunch controller -----------------------------------------------
echo "==> Relaunching controller..."
ros2 launch av_control controller.launch.py &
CTRL_PID=$!

# ---- Summary -----------------------------------------------------------
echo ""
echo "==> Run started."
echo "    Bag         : $BAG_PATH"
echo "    spawn PID   : $SPAWN_PID"
echo "    controller  : $CTRL_PID"
echo "    bag PID     : $BAG_PID"
echo ""
echo "    Press Ctrl+C to stop this run and save the bag."

# Stop everything cleanly on Ctrl+C
trap "echo ''; echo '==> Stopping run...'; \
      kill $CTRL_PID $SPAWN_PID $BAG_PID 2>/dev/null; \
      echo '==> Bag saved to $BAG_PATH'; \
      echo '==> Run stopped.'" INT

wait