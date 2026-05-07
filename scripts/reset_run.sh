#!/usr/bin/env bash
# scripts/reset_run.sh

set -e

source carla_ros_ws/install/setup.bash

mkdir -p bags

# ---- Stop everything from the previous run -----------------------------
echo "==> Stopping previous run..."
pkill -f "road_graph_planner"       2>/dev/null || true   
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
    /planning/trajectory \
    &
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

# ---- Relaunch planner --------------------------------------------------
echo "==> Relaunching planner..."                          
ros2 launch av_planning planner.launch.py &
PLANNER_PID=$!

echo "==> Waiting for planner to connect to CARLA (4 s)..."
sleep 4

# ---- Relaunch controller -----------------------------------------------
echo "==> Relaunching controller..."
ros2 launch av_control controller.launch.py &
CTRL_PID=$!

# ---- Summary -----------------------------------------------------------
echo ""
echo "==> Run started."
echo "    Bag         : $BAG_PATH"
echo "    spawn PID   : $SPAWN_PID"
echo "    planner PID : $PLANNER_PID"      
echo "    controller  : $CTRL_PID"
echo "    bag PID     : $BAG_PID"
echo ""
echo "    Press Ctrl+C to stop this run and save the bag."

trap "echo ''; echo '==> Stopping run...'; \
      kill $CTRL_PID $PLANNER_PID $SPAWN_PID $BAG_PID 2>/dev/null; \
      echo '==> Bag saved to $BAG_PATH'; \
      echo '==> Run stopped.'" INT

wait