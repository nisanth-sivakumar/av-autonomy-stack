#!/usr/bin/env bash
# scripts/reset_run.sh
# ---------------------
# Full pipeline launcher: bag → spawn → perception → planner → controller
#
# Usage:
#   ./scripts/reset_run.sh                  # road_graph planner (default)
#   ./scripts/reset_run.sh lattice          # lattice planner
#   ./scripts/reset_run.sh mppi             # road-constrained MPPI
#   ./scripts/reset_run.sh mppi -45 60      # ...and drive to goal (x, y) in the ROS map
#                                           # frame (A* route, stops there)
#   EGO_SPAWN="x,y,z,roll,pitch,yaw" ./scripts/reset_run.sh lattice
#                                           # spawn elsewhere (ROS frame, as in objects.json)
#   CONTROLLER=mpc ./scripts/reset_run.sh mppi
#                                           # LTV MPC (av_control_cpp) instead of Pure Pursuit
#
# Prerequisites: CARLA server running (Windows) and carla_ros_bridge running
# in another terminal. Can be run from any directory.
#
# Each stage waits for the previous one to actually produce data (with a
# timeout) instead of sleeping a fixed time.

set -eo pipefail
# Job control: without it, bash starts every `cmd &` with SIGINT *ignored*,
# so the nodes could never be stopped gracefully (spawn_objects would not
# destroy its ego — the source of the "two egos" runs). With -m each
# background job keeps normal SIGINT handling in its own process group, and
# Ctrl+C reaches only this script, whose trap stops the jobs in order.
set -m

PLANNER_TYPE="${1:-road_graph}"
case "$PLANNER_TYPE" in
    road_graph|lattice|mppi) ;;
    *) echo "Unknown planner '$PLANNER_TYPE' (expected road_graph | lattice | mppi)"; exit 1 ;;
esac
CONTROLLER="${CONTROLLER:-pure_pursuit}"
case "$CONTROLLER" in
    pure_pursuit|mpc) ;;
    *) echo "CONTROLLER must be pure_pursuit or mpc (got '$CONTROLLER')"; exit 1 ;;
esac
GOAL_ARGS=()
if [ $# -ge 3 ]; then
    GOAL_ARGS=("goal_x:=$2" "goal_y:=$3")
elif [ $# -eq 2 ]; then
    echo "Goal needs both coordinates: $0 <planner> <goal_x> <goal_y>"; exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source carla_ros_ws/install/setup.bash
# Unbuffered node output, so log lines appear in the order things happen
export PYTHONUNBUFFERED=1 RCUTILS_LOGGING_BUFFERED_STREAM=0

# ---- Helpers -------------------------------------------------------------

# wait_for_msg <topic> <msg_type> <timeout_s>: true once one message arrives.
# The type must be given: without it, Humble's `ros2 topic echo` exits
# immediately ("could not determine the type") if the topic doesn't exist
# *yet*, instead of waiting for it to appear.
wait_for_msg() {
    timeout "$3" ros2 topic echo --once "$1" "$2" >/dev/null 2>&1
}

# any_alive <pid>...: true if any of the processes still exists
any_alive() {
    local pid
    for pid in "$@"; do kill -0 "$pid" 2>/dev/null && return 0; done
    return 1
}

# stop_pids <pid>...: SIGINT first (spawn_objects destroys its actors, the
# bag is finalised), then SIGTERM after 15 s and SIGKILL after 5 more.
# Escalation is needed: carla_spawn_objects hangs in shutdown *after* it
# has destroyed its actors.
stop_pids() {
    [ $# -eq 0 ] && return 0
    kill -INT "$@" 2>/dev/null || true
    for _ in $(seq 30); do any_alive "$@" || return 0; sleep 0.5; done
    kill -TERM "$@" 2>/dev/null || true
    for _ in $(seq 10); do any_alive "$@" || return 0; sleep 0.5; done
    kill -KILL "$@" 2>/dev/null || true
}

# stop_matching <pattern>...: same escalation for processes found by name
# (previous runs, and nodes orphaned when their launch was killed).
stop_matching() {
    local pat pids=()
    for pat in "$@"; do
        while read -r pid; do [ -n "$pid" ] && pids+=("$pid"); done < <(pgrep -f "$pat" || true)
    done
    stop_pids "${pids[@]}"
}

# ego_id: id of the current ego from the bridge's latched vehicle_info ("" if none)
ego_id() {
    timeout 5 ros2 topic echo --once --qos-durability transient_local --qos-reliability reliable \
        /carla/ego_vehicle/vehicle_info carla_msgs/msg/CarlaEgoVehicleInfo 2>/dev/null \
        | awk '/^id:/ {print $2; exit}'
}

# destroy_ego: destroy the ego through the bridge's /carla/destroy_object service.
# The bridge removes the vehicle *and all its children*, including its
# bridge-side pseudo-sensors (odometry, tf, speedometer, control).
# carla_spawn_objects normally does this in its own shutdown, but in ROS 2
# that is unreliable: it can destroy the CARLA vehicle and then hang, leaving
# "phantom" pseudo-sensors that keep publishing a frozen ego pose.
destroy_ego() {
    local id
    id="$(ego_id)"
    [ -z "$id" ] && return 0
    echo "    destroying ego (id $id) via /carla/destroy_object"
    timeout 15 ros2 service call /carla/destroy_object carla_msgs/srv/DestroyObject \
        "{id: $id}" >/dev/null 2>&1 || true
    # wait until the bridge has processed it (odometry stops)
    for _ in $(seq 10); do
        wait_for_msg /carla/ego_vehicle/odometry nav_msgs/msg/Odometry 1 || return 0
    done
    echo "    WARNING: ego odometry still publishing after destroy"
}

# abort <msg>: stop whatever this run has started, then exit 1
abort() {
    echo "ERROR: $1"
    stop_pids ${CTRL_PID:-} ${PLANNER_PID:-} ${PERCEP_PID:-}
    destroy_ego
    stop_pids ${SPAWN_PID:-} ${BAG_PID:-}
    stop_matching "${NODE_PATTERNS[@]}"
    exit 1
}

NODE_PATTERNS=(
    "lib/av_control/controller_node"
    "lib/av_control_cpp/mpc_controller_node"
    "lib/av_planning/planner_node"
    "lib/av_perception/perception_node"
    "carla_spawn_objects"
    "ros2 bag record"
)

# ---- Preconditions -------------------------------------------------------

echo "==> Checking carla_ros_bridge..."
if ! ros2 topic list 2>/dev/null | grep -qx "/carla/status"; then
    echo "ERROR: /carla/status not found — is carla_ros_bridge running?"
    echo "  ros2 launch carla_ros_bridge carla_ros_bridge.launch.py host:=172.25.80.1 port:=2000 town:=Town10HD_Opt"
    exit 1
fi

# ---- Stop everything from the previous run -------------------------------

echo "==> Stopping previous run..."
stop_matching "${NODE_PATTERNS[@]}"

# A leftover ego (from a run that was killed hard, or bridge-side phantom
# pseudo-sensors) makes /carla/ego_vehicle/* carry two vehicles — the
# odometry then jumps between them and every metric is garbage.
echo "==> Checking no ego_vehicle is left over..."
if wait_for_msg /carla/ego_vehicle/odometry nav_msgs/msg/Odometry 3; then
    echo "    leftover ego found — removing it"
    destroy_ego
    if wait_for_msg /carla/ego_vehicle/odometry nav_msgs/msg/Odometry 3; then
        echo "ERROR: an ego_vehicle still publishes after destroying it."
        echo "  Restart the bridge (clears all bridge-side objects), then retry."
        exit 1
    fi
fi

# ---- Start bag recording -------------------------------------------------

mkdir -p bags
CTRL_TAG=""; [ "$CONTROLLER" = mpc ] && CTRL_TAG="_mpc"
RUN_ID="run_${PLANNER_TYPE}${CTRL_TAG}_$(date +%Y%m%d_%H%M%S)"
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
    /perception/objects \
    /planning/behavior \
    /control/compute_time_ms \
    /perception/lane_offset &
BAG_PID=$!

# ---- Spawn ego from config/objects.json ----------------------------------
# Passed explicitly: the launch file's default is the *installed* copy, which
# only changes when carla_spawn_objects is rebuilt.

SPAWN_ARGS=()
if [ -n "${EGO_SPAWN:-}" ]; then
    SPAWN_ARGS=("spawn_point_ego_vehicle:=$EGO_SPAWN")
    echo "==> Spawning objects (config/objects.json, ego at $EGO_SPAWN)..."
else
    echo "==> Spawning objects (config/objects.json)..."
fi
ros2 launch carla_spawn_objects carla_spawn_objects.launch.py \
    objects_definition_file:="$REPO_ROOT/config/objects.json" "${SPAWN_ARGS[@]}" &
SPAWN_PID=$!
if ! wait_for_msg /carla/ego_vehicle/odometry nav_msgs/msg/Odometry 30; then
    abort "no ego odometry 30 s after spawning (is the bridge ticking? see the bridge terminal)."
fi
echo "    ego odometry OK"

# ---- Perception ----------------------------------------------------------

echo "==> Launching perception..."
ros2 launch av_perception perception.launch.py &
PERCEP_PID=$!
if wait_for_msg /perception/obstacles geometry_msgs/msg/PoseArray 20; then
    echo "    /perception/obstacles OK"
else
    echo "    WARNING: no /perception/obstacles after 20 s (CARLA API connection?) — continuing"
fi

# ---- Planner -------------------------------------------------------------

echo "==> Launching planner (type: $PLANNER_TYPE)..."
ros2 launch av_planning planner.launch.py planner_type:="$PLANNER_TYPE" "${GOAL_ARGS[@]}" &
PLANNER_PID=$!
if ! wait_for_msg /planning/trajectory nav_msgs/msg/Path 30; then
    abort "no /planning/trajectory after 30 s — check the planner output above."
fi
echo "    /planning/trajectory OK"

# ---- Controller ----------------------------------------------------------

echo "==> Launching controller ($CONTROLLER)..."
ros2 launch av_control controller.launch.py controller:="$CONTROLLER" &
CTRL_PID=$!

# ---- Summary -------------------------------------------------------------

cat <<EOF

==> Run started.
    Planner     : $PLANNER_TYPE${GOAL_ARGS:+   goal: ${GOAL_ARGS[*]}}
    Bag         : $BAG_PATH
    PIDs        : spawn=$SPAWN_PID perception=$PERCEP_PID planner=$PLANNER_PID controller=$CTRL_PID bag=$BAG_PID

    Press Ctrl+C to stop this run and save the bag.
    Then:  python3 scripts/analyze_run.py $BAG_PATH
EOF

stop_run() {
    trap - INT TERM
    echo ""
    echo "==> Stopping run..."
    ros2 topic pub --once /carla/ego_vehicle/vehicle_control_cmd \
        carla_msgs/msg/CarlaEgoVehicleControl \
        '{throttle: 0.0, steer: 0.0, brake: 1.0}' >/dev/null 2>&1 || true
    # Stack first, then the ego (explicitly, see destroy_ego), then
    # spawn_objects and the bag
    stop_pids $CTRL_PID $PLANNER_PID $PERCEP_PID
    destroy_ego
    stop_pids $SPAWN_PID $BAG_PID
    stop_matching "${NODE_PATTERNS[@]}"   # orphans, if a launch had to be killed
    echo "==> Bag saved to $BAG_PATH"
    echo "==> Run stopped."
    exit 0
}
trap stop_run INT TERM

wait
