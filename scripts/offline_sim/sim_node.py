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

At the end it writes a JSON report with ground-truth metrics:
cross-track error vs lane centre, distance travelled, minimum distance to
obstacles, collisions, and perception lane-offset error.

Usage:  python3 sim_node.py --duration 35 --out result.json
(PYTHONPATH must contain scripts/offline_sim/fake_carla.)
"""

import argparse
import json
import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
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
# Collision if the two footprints overlap (road frame, both cars aligned
# with the lane): Model 3 is 4.7 × 1.85 m, the obstacle car ~4.0 × 1.7 m.
COLLISION_LONG = (4.7 + 4.0) / 2
COLLISION_LAT = (1.85 + 1.7) / 2


class SimNode(Node):
    def __init__(self, duration, start_s):
        super().__init__("offline_sim")
        self.duration = duration
        x, y, yaw = road.pose_at(start_s)
        self.x, self.y, self.yaw, self.v = x, y, yaw, 0.0
        self.cmd = (0.0, 0.0, 0.0)
        self.t = 0.0
        self.obstacles = road.load_obstacles()
        self.obstacles_sl = [road.project(ox, oy) for ox, oy in self.obstacles]

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
        self.img_pub = self.create_publisher(
            Image, "/carla/ego_vehicle/semantic_segmentation_front/image", 1)
        self.info_pub = self.create_publisher(
            CameraInfo, "/carla/ego_vehicle/semantic_segmentation_front/camera_info", 1)
        self.create_subscription(
            CarlaEgoVehicleControl, "/carla/ego_vehicle/vehicle_control_cmd", self._cmd_cb, 10)
        self.create_subscription(Float32, "/perception/lane_offset", self._lane_cb, 10)
        self.create_subscription(Path, "/planning/trajectory", self._traj_cb, 1)

        self.create_timer(DT, self._step)
        self.create_timer(0.1, self._publish_camera)
        self.done = False

    # ── inputs ──────────────────────────────────────────────────────────
    def _cmd_cb(self, msg):
        self.cmd = (msg.throttle, msg.steer, msg.brake)

    def _lane_cb(self, msg):
        self.lane_offset_msgs += 1
        # ground truth: + = ego left of centre
        self.lane_err.append(msg.data - self.last_lateral)

    def _traj_cb(self, msg):
        self.traj_msgs += 1

    # ── dynamics + metrics ───────────────────────────────────────────────
    def _step(self):
        if self.done:
            return
        if self.traj_msgs == 0:
            # Hold time until the stack is up (first trajectory received)
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

        s, lateral = road.project(self.x, self.y)
        self.last_lateral = lateral
        self.final_s = s
        if self.t > 3.0:
            self.cte.append(lateral)
        self.max_speed = max(self.max_speed, self.v)
        if self.v < 0.05:
            self.stopped_time += DT
        if self.obstacles:
            d = min(math.hypot(self.x - ox, self.y - oy) for ox, oy in self.obstacles)
            self.min_obs = min(self.min_obs, d)
            hit = any(abs(s - os_) < COLLISION_LONG and abs(lateral - ol) < COLLISION_LAT
                      for os_, ol in self.obstacles_sl)
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
        for lat in (-road.LANE_WIDTH / 2, road.LANE_WIDTH / 2):
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
            "collisions": self.collisions,
            "stopped_time_s": round(self.stopped_time, 2),
            "trajectory_msgs": self.traj_msgs,
            "lane_offset_msgs": self.lane_offset_msgs,
            "lane_offset_mean_abs_err_m": None if lane is None else round(float(lane.mean()), 3),
            "lane_offset_p90_abs_err_m": None if lane is None else round(float(np.percentile(lane, 90)), 3),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=35.0)
    ap.add_argument("--start-s", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rclpy.init()
    node = SimNode(args.duration, args.start_s)
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
