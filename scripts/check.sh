#!/usr/bin/env bash
# scripts/check.sh — the project's test gate (local stand-in for CI; CI calls this too)
#
#   ./scripts/check.sh            # build + unit tests + offline scenarios, both controllers (~9 min)
#   ./scripts/check.sh --quick    # build + unit tests only
#   CHECK_RATE=2 ./scripts/check.sh   # offline sim speed-up (default 4; CI uses less on small runners)
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
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg (expected --quick)"; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGES=(av_control av_control_cpp av_planning av_perception)
CHECK_OUT="${CHECK_OUT:-$REPO_ROOT/log/check/offline_sim}"
CHECK_RATE="${CHECK_RATE:-4}"

step() { echo; echo "==> $*"; }

# fail TITLE FILE: report a failing step and exit.  On GitHub Actions the tail
# of FILE also goes into an error annotation, shown on the run's summary page
# (and readable through the public API, unlike the job log).
fail() {
    echo "FAIL: $1"
    if [ -n "${GITHUB_ACTIONS:-}" ] && [ -f "$2" ]; then
        local msg
        msg=$(tail -n 60 "$2" | sed -e 's/%/%25/g' | sed -e ':a;N;$!ba;s/\n/%0A/g')
        echo "::error title=check.sh: $1::$msg"
    fi
    exit 1
}
LOGDIR="$(mktemp -d)"

# ROS environment (the setup scripts reference unset variables, so no `set -u`)
if [ -z "${ROS_DISTRO:-}" ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi

cd "$REPO_ROOT/carla_ros_ws"

step "Build (${PACKAGES[*]} and their dependencies)"
if ! colcon build --symlink-install --packages-up-to "${PACKAGES[@]}" \
        --event-handlers console_cohesion+ > "$LOGDIR/build.log" 2>&1; then
    tail -n 80 "$LOGDIR/build.log"
    fail "build" "$LOGDIR/build.log"
fi
tail -n 5 "$LOGDIR/build.log"
# shellcheck disable=SC1091
source install/setup.bash

step "Unit tests"
# colcon test exits 0 on test failures (test-result catches those), non-zero
# when a test step itself errors
if ! colcon test --packages-select "${PACKAGES[@]}" --event-handlers console_cohesion+ \
        > "$LOGDIR/test.log" 2>&1; then
    tail -n 80 "$LOGDIR/test.log"
    fail "colcon test" "$LOGDIR/test.log"
fi
if ! colcon test-result --verbose --test-result-base build > "$LOGDIR/result.log" 2>&1; then
    cat "$LOGDIR/result.log"
    fail "unit tests" "$LOGDIR/result.log"
fi
tail -n 1 "$LOGDIR/result.log"

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
    if ! python3 scripts/offline_sim/run_scenarios.py --controller "$1" --vehicle "$2" --rate "$CHECK_RATE" \
            --workdir "$out" | tee "$out/summary.txt"; then
        fail "offline scenarios, $1 (see $out)" "$out/summary.txt"
    fi
done

echo; echo "==> check passed"
