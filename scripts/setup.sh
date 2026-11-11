#!/usr/bin/env bash
# scripts/setup.sh — one-time setup after cloning (the README and CI both use it)
#
#   ./scripts/setup.sh            # submodule + bridge patch + ROS dependencies (rosdep)
#   ./scripts/setup.sh --analysis # also the bag-analysis Python packages (pip, --user)
#
# Then build and test with ./scripts/check.sh.
#
# Needs ROS 2 Humble (/opt/ros/humble) and sudo for apt (rosdep). Safe to re-run:
# the patch is only applied if it isn't already.  CARLA itself (server and the
# 0.9.15 Python client) is only needed to drive in the simulator, not to build,
# test or run the offline scenarios.

set -eo pipefail

ANALYSIS=0
for arg in "$@"; do
    case "$arg" in
        --analysis) ANALYSIS=1 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg (expected --analysis)"; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRIDGE="$REPO_ROOT/carla_ros_ws/src/ros-bridge"
PATCH="$REPO_ROOT/patches/ros-bridge-local.patch"

step() { echo; echo "==> $*"; }

# root in a container (CI) has no sudo
SUDO=sudo
if [ "$(id -u)" = 0 ]; then SUDO=""; fi

if [ -z "${ROS_DISTRO:-}" ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi

step "ros-bridge submodule"
git -C "$REPO_ROOT" submodule update --init --recursive

step "Local ros-bridge patch"
if git -C "$BRIDGE" apply --check "$PATCH" 2>/dev/null; then
    git -C "$BRIDGE" apply "$PATCH"
    echo "applied"
elif git -C "$BRIDGE" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "already applied"
else
    echo "FAIL: $PATCH neither applies nor is applied (submodule at an unexpected commit?)"
    exit 1
fi

step "ROS dependencies (rosdep)"
if [ ! -e /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    $SUDO rosdep init
fi
rosdep update --rosdistro humble
# Only this project's packages: the rest of ros-bridge isn't built (check.sh
# builds --packages-up-to), and some of its packages have keys rosdep can't resolve.
cd "$REPO_ROOT/carla_ros_ws"
PATHS=(src/av_control src/av_control_cpp src/av_planning src/av_perception src/ros-bridge/carla_msgs)
if rosdep check --rosdistro humble --from-paths "${PATHS[@]}" --ignore-src; then
    echo "all ROS dependencies installed"
else
    $SUDO apt-get update -qq
    rosdep install -y --rosdistro humble --from-paths "${PATHS[@]}" --ignore-src
fi

if [ "$ANALYSIS" = 1 ]; then
    step "Bag-analysis packages (scripts/analyze_run.py)"
    pip3 install --user rosbags matplotlib pandas "numpy<2"
fi

echo; echo "==> setup done; now ./scripts/check.sh"
