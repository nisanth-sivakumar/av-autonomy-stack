#!/usr/bin/env python3
"""
Offline stand-in for CARLA + carla_ros_bridge, plus a test monitor.

Simulates the ego vehicle with a kinematic bicycle model driven by
/carla/ego_vehicle/vehicle_control_cmd and publishes what the bridge would:

    /carla/ego_vehicle/odometry                                (nav_msgs/Odometry)
    /carla/ego_vehicle/semantic_segmentation_front/image       (sensor_msgs/Image, bgra8 palette)
    /carla/ego_vehicle/semantic_segmentation_front/camera_info (sensor_msgs/CameraInfo)

The camera image is rendered from the true lane geometry (offline_road) with
the same intrinsics/mount as config/objects.json (400×200, fov 90, x=2, z=2,
pitched 20° down).

It also publishes /clock (the stack runs on simulated time).  The clock
runs from the start; the vehicle is held until the first trajectory
arrives.  From then on the drive time (simulated seconds) is written to
$OFFLINE_SIM_CLOCK every step: the time base for moving scenario
obstacles in every process (see offline_road.py).  --rate N steps the
simulation N× faster than real time; the nodes follow /clock.

At the end it writes a JSON report with ground-truth metrics:
cross-track error vs lane centre, distance travelled, minimum distance to
obstacles, collisions, distance to the goal, and perception lane-offset
error.

Usage:  python3 sim_node.py --duration 35 --out result.json
(PYTHONPATH must contain scripts/offline_sim/fake_carla.)
"""

import argparse
import json
import math
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32, String
from rosgraph_msgs.msg import Clock
from carla_msgs.msg import CarlaEgoVehicleControl

import offline_road as road

DT = 0.05
WHEELBASE = 2.875
MAX_WHEEL_ANGLE = math.radians(70.0)
CAM_W, CAM_H, CAM_FOV = 400, 200, 90.0
CAM_X, CAM_Z = 2.0, 2.0
CAM_PITCH = math.radians(20.0)      # looking down, as in config/objects.json
ROAD_LINE_BGRA = (50, 234, 157, 255)
ROAD_BGRA = (128, 64, 128, 255)
# Collision if the footprints overlap (road frame, both cars aligned with
# the lane).  Model 3: 4.7 × 1.85 m; obstacle size from the scenario.
EGO_HALF_LENGTH, EGO_HALF_WIDTH = 4.7 / 2, 1.85 / 2


class SimNode(Node):
    def __init__(self, duration, start_s, goal_s=None, start_lateral=0.0,
                 start_yaw_deg=0.0, rate=1.0):
        super().__init__("offline_sim")
        self.duration = duration
        x, y = road.point_at(start_s, start_lateral)
        yaw = road.pose_at(start_s)[2] + math.radians(start_yaw_deg)
        self.x, self.y, self.yaw, self.v = x, y, yaw, 0.0
        self.cmd = (0.0, 0.0, 0.0)
        self.t = 0.0
        self.obstacle_specs = road.obstacle_specs()
        self.goal_s = goal_s
        self.clock_t = 0.0
        self.clock_path = os.environ.get("OFFLINE_SIM_CLOCK")
        self.trace = []                          # [t, s, lateral, v] every 0.5 s
        self.steps = 0
        self.trigger_path = os.environ.get("OFFLINE_SIM_TRIGGER")
        self.triggered = False
        self.max_lateral = 0.0
        self.min_side_clearance = float("inf")   # body gap to cars alongside in other lanes
        self.behavior_states: list[str] = []      # state sequence (changes only)
        self.lane_changes = 0
        self.aborts = 0
        self.min_gap_lead = float("inf")      # bumper gap to obstacles ahead in the lane

        self.f = CAM_W / (2.0 * math.tan(math.radians(CAM_FOV) / 2.0))
        self.cx, self.cy = CAM_W / 2.0, CAM_H / 2.0

        # metrics
        self.cte = []
        self.lane_err = []
        self.min_obs = float("inf")
        self.collisions = 0
        self.in_collision = False
        self.max_speed = 0.0
        self.lane_offset_msgs = 0
        self.traj_msgs = 0
        self.last_lateral = 0.0
        self.final_s = start_s
        self.stopped_time = 0.0

        self.odom_pub = self.create_publisher(Odometry, "/carla/ego_vehicle/odometry", 10)
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.img_pub = self.create_publisher(
            Image, "/carla/ego_vehicle/semantic_segmentation_front/image", 1)
        self.info_pub = self.create_publisher(
            CameraInfo, "/carla/ego_vehicle/semantic_segmentation_front/camera_info", 1)
        self.create_subscription(
            CarlaEgoVehicleControl, "/carla/ego_vehicle/vehicle_control_cmd", self._cmd_cb, 10)
        self.create_subscription(Float32, "/perception/lane_offset", self._lane_cb, 10)
        self.create_subscription(Path, "/planning/trajectory", self._traj_cb, 1)
        self.create_subscription(String, "/planning/behavior", self._behavior_cb, 10)

        # wall period DT / rate: the stack follows /clock, so it speeds up too
        self.create_timer(DT / max(rate, 1e-3), self._step)
        self.done = False

    # ── inputs ──────────────────────────────────────────────────────────
    def _cmd_cb(self, msg):
        self.cmd = (msg.throttle, msg.steer, msg.brake)

    def _lane_cb(self, msg):
        self.lane_offset_msgs += 1
        # ground truth: + = ego left of centre
        self.lane_err.append(msg.data - self.last_lateral)

    def _behavior_cb(self, msg):
        state = msg.data.split()[0].split("=", 1)[-1]
        if not self.behavior_states or self.behavior_states[-1] != state:
            if state == "LANE_KEEP" and self.behavior_states and self.behavior_states[-1] == "CHANGING":
                self.lane_changes += 1
            if state == "ABORT":
                self.aborts += 1
            self.behavior_states.append(state)

    def _traj_cb(self, msg):
        self.traj_msgs += 1

    # ── dynamics + metrics ───────────────────────────────────────────────
    def _publish_clock(self):
        self.clock_t += DT
        msg = Clock()
        msg.clock.sec = int(self.clock_t)
        msg.clock.nanosec = int((self.clock_t % 1.0) * 1e9)
        self.clock_pub.publish(msg)

    def _step(self):
        if self.done:
            return
        self._publish_clock()
        self.steps += 1
        if self.steps % 2 == 0:
            self._publish_camera()               # 10 Hz of simulated time
        if self.traj_msgs == 0:
            # Hold the vehicle until the stack is up (first trajectory received)
            self._publish_odom()
            return

        throttle, steer, brake = self.cmd
        acc = 3.0 * throttle - 7.0 * brake - 0.05 * self.v
        self.v = max(0.0, self.v + acc * DT)
        delta = -steer * MAX_WHEEL_ANGLE          # CARLA +steer = right turn
        self.yaw += self.v / WHEELBASE * math.tan(delta) * DT
        self.x += self.v * math.cos(self.yaw) * DT
        self.y += self.v * math.sin(self.yaw) * DT
        self.t += DT
        if self.clock_path:                      # shared time base for moving obstacles
            tmp = self.clock_path + ".tmp"
            with open(tmp, "w") as f:
                f.write(repr(self.t))
            os.replace(tmp, self.clock_path)

        s, lateral = road.project(self.x, self.y)
        # perception measures the offset from the lane the car is in
        nearest = min(road.LANES.values(), key=lambda l: abs(l - lateral))
        self.last_lateral = lateral - nearest
        self.max_lateral = max(self.max_lateral, abs(lateral))
        if not self.triggered and abs(lateral) > 0.5 and self.trigger_path:
            self.triggered = True                     # the ego started leaving its lane
            with open(self.trigger_path, "w") as f:
                f.write(repr(self.t))
        self.final_s = s
        if self.steps % 10 == 0:                 # coarse trace for diagnosing failures
            self.trace.append([round(self.t, 1), round(s, 1), round(lateral, 2),
                               round(self.v, 2)])
        if self.t > 3.0:
            self.cte.append(lateral)
        self.max_speed = max(self.max_speed, self.v)
        if self.v < 0.05:
            self.stopped_time += DT
        if self.obstacle_specs:
            hit = False
            for spec in self.obstacle_specs:
                st = road.obstacle_now(spec)
                self.min_obs = min(self.min_obs, math.hypot(self.x - st["x"], self.y - st["y"]))
                ds, dl = st["s"] - s, st["lateral"] - lateral
                if abs(ds) < EGO_HALF_LENGTH + spec["half_length"] and \
                        abs(dl) < EGO_HALF_WIDTH + spec["half_width"]:
                    hit = True
                if ds > 0 and abs(dl) < EGO_HALF_WIDTH + spec["half_width"]:
                    self.min_gap_lead = min(
                        self.min_gap_lead, ds - EGO_HALF_LENGTH - spec["half_length"])
                if abs(ds) < EGO_HALF_LENGTH + spec["half_length"]:
                    self.min_side_clearance = min(
                        self.min_side_clearance, abs(dl) - EGO_HALF_WIDTH - spec["half_width"])
            if hit and not self.in_collision:
                self.collisions += 1
            self.in_collision = hit

        self._publish_odom()
        if self.t >= self.duration:
            self.done = True

    def _publish_odom(self):
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id = "ego_vehicle"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = math.sin(self.yaw / 2)
        odom.pose.pose.orientation.w = math.cos(self.yaw / 2)
        odom.twist.twist.linear.x = self.v
        self.odom_pub.publish(odom)

    # ── camera ──────────────────────────────────────────────────────────
    def _project(self, px, py):
        cam_x = self.x + CAM_X * math.cos(self.yaw)
        cam_y = self.y + CAM_X * math.sin(self.yaw)
        dx, dy = px - cam_x, py - cam_y
        fwd = dx * math.cos(self.yaw) + dy * math.sin(self.yaw)
        left = -dx * math.sin(self.yaw) + dy * math.cos(self.yaw)
        # rotate into the pitched camera frame (z forward, y down, x right)
        zc = fwd * math.cos(CAM_PITCH) + CAM_Z * math.sin(CAM_PITCH)
        yc = -fwd * math.sin(CAM_PITCH) + CAM_Z * math.cos(CAM_PITCH)
        if fwd < 0.3 or zc < 0.1:
            return None
        return (self.cx + self.f * (-left) / zc, self.cy + self.f * yc / zc)

    def _publish_camera(self):
        img = np.zeros((CAM_H, CAM_W, 4), np.uint8)
        img[:, :] = ROAD_BGRA
        s0, _ = road.project(self.x, self.y)
        boundaries = sorted({l + sgn * road.LANE_WIDTH / 2
                             for l in road.LANES.values() for sgn in (-1, 1)})
        for lat in boundaries:
            pts = []
            for ds in np.arange(-2.0, 60.0, 0.2):
                p = self._project(*road.point_at(s0 + ds, lat))
                if p is not None:
                    pts.append(p)
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(img, (int(round(a[0])), int(round(a[1]))),
                         (int(round(b[0])), int(round(b[1]))), ROAD_LINE_BGRA, 2)
        stamp = self.get_clock().now().to_msg()
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "ego_vehicle/semantic_segmentation_front"
        msg.height, msg.width = CAM_H, CAM_W
        msg.encoding = "bgra8"
        msg.step = CAM_W * 4
        msg.data = img.tobytes()
        self.img_pub.publish(msg)

        info = CameraInfo()
        info.header = msg.header
        info.width, info.height = CAM_W, CAM_H
        info.k = [self.f, 0.0, self.cx, 0.0, self.f, self.cy, 0.0, 0.0, 1.0]
        self.info_pub.publish(info)

    # ── report ──────────────────────────────────────────────────────────
    def report(self):
        cte = np.abs(np.asarray(self.cte)) if self.cte else np.zeros(1)
        lane = np.abs(np.asarray(self.lane_err)) if self.lane_err else None
        return {
            "sim_time_s": round(self.t, 2),
            "final_s_m": round(self.final_s, 2),
            "max_speed_mps": round(self.max_speed, 2),
            "final_speed_mps": round(self.v, 2),
            "mean_abs_cte_m": round(float(cte.mean()), 3),
            "max_abs_cte_m": round(float(cte.max()), 3),
            "min_obstacle_dist_m": None if math.isinf(self.min_obs) else round(self.min_obs, 2),
            "min_lead_gap_m": None if math.isinf(self.min_gap_lead) else round(self.min_gap_lead, 2),
            "trace": self.trace,
            "obstacle_final_s_m": [round(road.obstacle_now(o)["s"], 2) for o in self.obstacle_specs],
            "goal_error_m": None if self.goal_s is None else round(self.final_s - self.goal_s, 2),
            "collisions": self.collisions,
            "stopped_time_s": round(self.stopped_time, 2),
            "trajectory_msgs": self.traj_msgs,
            "lane_offset_msgs": self.lane_offset_msgs,
            "final_lateral_m": round(road.project(self.x, self.y)[1], 2),
            "max_lateral_m": round(self.max_lateral, 2),
            "min_side_clearance_m": None if math.isinf(self.min_side_clearance)
            else round(self.min_side_clearance, 2),
            "lane_changes": self.lane_changes,
            "aborts": self.aborts,
            "behavior_states": self.behavior_states[-12:],
            "lane_offset_mean_abs_err_m": None if lane is None else round(float(lane.mean()), 3),
            "lane_offset_p90_abs_err_m": None if lane is None else round(float(np.percentile(lane, 90)), 3),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=35.0)
    ap.add_argument("--start-s", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--goal-s", type=float, default=None, help="goal arc length (for the report)")
    ap.add_argument("--start-lateral", type=float, default=0.0, help="start offset, m (+ left)")
    ap.add_argument("--start-yaw-deg", type=float, default=0.0, help="start heading error")
    ap.add_argument("--rate", type=float, default=1.0, help="simulated seconds per wall second")
    args = ap.parse_args()

    rclpy.init()
    node = SimNode(args.duration, args.start_s, args.goal_s, args.start_lateral,
                   args.start_yaw_deg, args.rate)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        with open(args.out, "w") as f:
            json.dump(node.report(), f, indent=2)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
