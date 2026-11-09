#!/usr/bin/env bash
# scripts/check.sh — the project's test gate (local stand-in for CI; CI calls this too)
#
#   ./scripts/check.sh            # build + unit tests + offline scenarios, both controllers (~9 min)
#   ./scripts/check.sh --quick    # build + unit tests only
#
# Steps:
#   1. colcon build of the stack and its dependencies (carla_msgs etc. from ros-bridge)
#   2. colcon test + colcon test-result --verbose (fails on any test failure)
#   3. unless --quick: scripts/offline_sim/run_scenarios.py for Pure Pursuit and the MPC
#      (fails if any scenario fails)
#
# Exits non-zero on the first failing step.  Offline-scenario logs and results
# go to $CHECK_OUT (default: log/check/offline_sim under the repo root), which
# CI uploads as artifacts.
#
# Don't run the full check while a CARLA run is going: two stacks at once
# stalled DDS.  The offline sim uses its own ROS_DOMAIN_ID (77) regardless.

set -eo pipefail

QUICK=0
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg (expected --quick)"; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGES=(av_control av_control_cpp av_planning av_perception)
CHECK_OUT="${CHECK_OUT:-$REPO_ROOT/log/check/offline_sim}"

step() { echo; echo "==> $*"; }

# ROS environment (the setup scripts reference unset variables, so no `set -u`)
if [ -z "${ROS_DISTRO:-}" ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi

cd "$REPO_ROOT/carla_ros_ws"

step "Build (${PACKAGES[*]} and their dependencies)"
colcon build --symlink-install --packages-up-to "${PACKAGES[@]}" \
    --event-handlers console_cohesion+ 2>&1 | tail -n 40
# shellcheck disable=SC1091
source install/setup.bash

step "Unit tests"
colcon test --packages-select "${PACKAGES[@]}" >/dev/null
# colcon test always exits 0; test-result exits non-zero on any failure
if ! colcon test-result --verbose --test-result-base build; then
    echo "FAIL: unit tests"
    exit 1
fi

if [ "$QUICK" = 1 ]; then
    echo; echo "==> check --quick passed"
    exit 0
fi

step "Offline closed-loop scenarios (results: $CHECK_OUT)"
mkdir -p "$CHECK_OUT"
cd "$REPO_ROOT"
# Both controllers, each in the sim geometry it currently passes: Pure
# Pursuit in the legacy model (its own), the MPC in CARLA's Model 3 geometry
# (rear-axle bicycle, 3.005 m).  --rate 4 gives the same results as 1x.
for ctl in "pure_pursuit legacy" "mpc carla"; do
    set -- $ctl
    out="$CHECK_OUT/$1"
    mkdir -p "$out"
    echo "--- controller $1, vehicle $2"
    if ! python3 scripts/offline_sim/run_scenarios.py --controller "$1" --vehicle "$2" --rate 4 \
            --workdir "$out" | tee "$out/summary.txt"; then
        echo "FAIL: offline scenarios, $1 (see $out)"
        exit 1
    fi
done

echo; echo "==> check passed"
