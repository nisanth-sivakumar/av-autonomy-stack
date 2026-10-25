#!/usr/bin/env python3
"""
perception_node.py — av_perception package
======================================================
Provides two perception capabilities:

  1. Object detection
     Queries the CARLA Python API for nearby vehicles and pedestrians,
     adds Gaussian noise to simulate real sensor uncertainty, and
     publishes obstacle positions in the ROS map frame.

  2. Lane detection
     Finds road-line pixels (palette colour) in the semantic segmentation
     camera image and projects them onto the ground plane using the
     camera intrinsics and mount height (see lane_detection.py) to get
     the ego's lateral offset from the lane centre in metres.

Publishes
---------
  /perception/obstacles       (geometry_msgs/PoseArray)
      Detected obstacle positions in map (ROS) frame.
      Positive x = forward, positive y = left.

  /perception/lane_offset     (std_msgs/Float32)
      Signed lateral offset from lane centre in metres.
      Positive = ego is to the LEFT of centre (needs to steer right).

  /perception/debug/image     (sensor_msgs/Image)
      Annotated semantic segmentation image for RViz / debugging.

Subscribes
----------
  /carla/ego_vehicle/semantic_segmentation_front/image        (sensor_msgs/Image)
  /carla/ego_vehicle/semantic_segmentation_front/camera_info  (sensor_msgs/CameraInfo)
  /carla/ego_vehicle/odometry                           (nav_msgs/Odometry)

Connects to CARLA Python API (read-only) to query actor positions.

Coordinate conventions
----------------------
    ros_x =  carla_x
    ros_y = -carla_y
    Obstacle positions published in ROS map frame.

The bridge publishes the semantic image already converted to the
CityScapes palette, so road lines are detected by colour (157, 234, 50),
not by label ID.
"""

import math
import time
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32

try:
    from cv_bridge import CvBridge
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

from av_perception.lane_detection import (
    CameraModel,
    road_line_mask,
    estimate_lane_offset,
)


# ---------------------------------------------------------------------------
# Perception node
# ---------------------------------------------------------------------------

class PerceptionNode(Node):

    def __init__(self):
        super().__init__("perception_node")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("carla_host",            "localhost")
        self.declare_parameter("carla_port",            2000)
        self.declare_parameter("carla_timeout",         10.0)

        # Object detection
        self.declare_parameter("detection_range_m",     50.0)   # metres ahead/around
        self.declare_parameter("min_detection_dist_m",  2.0)    # ignore closer than this
        self.declare_parameter("position_noise_std_m",  0.4)    # Gaussian noise on position
        self.declare_parameter("detection_rate_hz",     10.0)

        # Lane detection — camera geometry must match config/objects.json
        self.declare_parameter("camera_height_m",       2.0)    # mount height above ground
        self.declare_parameter("camera_pitch_deg",      20.0)   # positive = looking down
        self.declare_parameter("camera_fov_deg",        90.0)   # used until camera_info arrives
        self.declare_parameter("lane_max_range_m",      15.0)   # ignore ground farther than this
        self.declare_parameter("lane_width_m",          3.5)    # assumed when one marking visible
        self.declare_parameter("lane_min_rows",         3)      # rows needed to trust an estimate
        self.declare_parameter("lane_hold_time_s",      1.0)    # hold last estimate this long
        self.declare_parameter("publish_debug_image",   True)

        # ── State ─────────────────────────────────────────────────────────
        self._ego_x:     float | None = None
        self._ego_y:     float | None = None
        self._ego_yaw:   float | None = None

        self._carla_world    = None
        self._ego_actor_id:  int | None = None
        self._obstacles:     list[tuple[float, float]] = []

        self._cv_bridge = CvBridge() if CV_AVAILABLE else None
        self._last_lane_offset: float | None = None
        self._last_lane_stamp: float = 0.0
        self._camera_info: CameraInfo | None = None

        # ── CARLA connection ──────────────────────────────────────────────
        self._connect_to_carla()

        # ── QoS ───────────────────────────────────────────────────────────
        odom_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        sensor_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ── Publishers ────────────────────────────────────────────────────
        self._obstacle_pub   = self.create_publisher(PoseArray, "/perception/obstacles",     10)
        self._lane_offset_pub = self.create_publisher(Float32,  "/perception/lane_offset",   10)
        self._debug_img_pub  = self.create_publisher(Image,     "/perception/debug/image",   1)

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            Odometry,
            "/carla/ego_vehicle/odometry",
            self._odom_cb,
            odom_qos,
        )
        self.create_subscription(
            Image,
            "/carla/ego_vehicle/semantic_segmentation_front/image",
            self._semantic_image_cb,
            sensor_qos,
        )
        self.create_subscription(
            CameraInfo,
            "/carla/ego_vehicle/semantic_segmentation_front/camera_info",
            self._camera_info_cb,
            sensor_qos,
        )

        # ── Timers ────────────────────────────────────────────────────────
        rate = self.get_parameter("detection_rate_hz").value
        self.create_timer(1.0 / rate, self._detect_objects_cb)

        self.get_logger().info("PerceptionNode ready — waiting for odometry and camera…")

    # ── CARLA connection ──────────────────────────────────────────────────

    def _connect_to_carla(self):
        if not CARLA_AVAILABLE:
            self.get_logger().error(
                "CARLA Python API not found — object detection disabled."
            )
            return

        host    = self.get_parameter("carla_host").value
        port    = self.get_parameter("carla_port").value
        timeout = self.get_parameter("carla_timeout").value

        try:
            client = carla.Client(host, port)
            client.set_timeout(timeout)
            self._carla_world = client.get_world()
            self.get_logger().info(
                f"CARLA connected — map={self._carla_world.get_map().name}"
            )
        except Exception as exc:
            self.get_logger().error(f"CARLA connection failed: {exc}")

    # ── Odometry callback ─────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self._ego_x   = msg.pose.pose.position.x
        self._ego_y   = msg.pose.pose.position.y
        q             = msg.pose.pose.orientation
        siny_cosp     = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp     = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._ego_yaw = math.atan2(siny_cosp, cosy_cosp)

    # ── Object detection (CARLA Python API) ───────────────────────────────

    def _detect_objects_cb(self):
        """
        Query CARLA for nearby actors, apply Gaussian position noise,
        and publish as PoseArray on /perception/obstacles.

        The noise simulates real sensor uncertainty.  Tune position_noise_std_m
        to match your target sensor (e.g., 0.2m for LiDAR, 0.5m for camera).
        """
        if self._carla_world is None or self._ego_x is None:
            return

        detection_range = self.get_parameter("detection_range_m").value
        min_dist        = self.get_parameter("min_detection_dist_m").value
        noise_std       = self.get_parameter("position_noise_std_m").value

        try:
            actors = self._carla_world.get_actors()
        except Exception as exc:
            self.get_logger().warn(f"get_actors failed: {exc}")
            return

        # Lazily cache ego vehicle actor ID so we can exclude it
        if self._ego_actor_id is None:
            for actor in actors.filter("vehicle.*"):
                attrs = actor.attributes
                if attrs.get("role_name") in ("ego_vehicle", "hero"):
                    self._ego_actor_id = actor.id
                    break

        detected = []
        # Check vehicles and pedestrians
        for filter_str in ("vehicle.*", "walker.pedestrian.*"):
            for actor in actors.filter(filter_str):
                if actor.id == self._ego_actor_id:
                    continue

                loc     = actor.get_transform().location
                ros_x   = loc.x
                ros_y   = -loc.y   # CARLA → ROS frame

                dist = math.hypot(ros_x - self._ego_x, ros_y - self._ego_y)
                if dist < min_dist or dist > detection_range:
                    continue

                # Gaussian noise simulates sensor imprecision
                noisy_x = ros_x + np.random.normal(0.0, noise_std)
                noisy_y = ros_y + np.random.normal(0.0, noise_std)
                detected.append((noisy_x, noisy_y))

        self._obstacles = detected

        # Publish
        msg = PoseArray()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for (ox, oy) in detected:
            p = Pose()
            p.position.x = ox
            p.position.y = oy
            p.position.z = 0.0
            p.orientation.w = 1.0
            msg.poses.append(p)

        self._obstacle_pub.publish(msg)

        if detected:
            self.get_logger().debug(
                f"[objects] {len(detected)} obstacles detected  "
                f"closest={min(math.hypot(ox-self._ego_x, oy-self._ego_y) for ox,oy in detected):.1f} m"
            )

    # ── Lane detection (semantic segmentation) ────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo):
        self._camera_info = msg

    def _camera_model(self, width: int, height: int) -> CameraModel:
        cam_h = self.get_parameter("camera_height_m").value
        pitch = self.get_parameter("camera_pitch_deg").value
        info = self._camera_info
        if info is not None and info.width == width and info.k[0] > 0.0:
            return CameraModel(fx=info.k[0], cx=info.k[2], cy=info.k[5], height_m=cam_h,
                               pitch_rad=math.radians(pitch))
        return CameraModel.from_fov(
            width, height, self.get_parameter("camera_fov_deg").value, cam_h, pitch)

    def _semantic_image_cb(self, msg: Image):
        """
        Estimate the lateral offset from the lane centre and publish it.

        Publishes /perception/lane_offset only while an estimate is
        available (fresh, or held for `lane_hold_time_s`), so consumers can
        tell "no detection" from "centred".
        Positive = ego LEFT of lane centre.
        """
        if not CV_AVAILABLE:
            return
        try:
            bgr = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        cam = self._camera_model(w, h)
        mask = road_line_mask(rgb)
        est = estimate_lane_offset(
            mask, cam,
            max_range_m=self.get_parameter("lane_max_range_m").value,
            lane_width_m=self.get_parameter("lane_width_m").value,
            min_rows=self.get_parameter("lane_min_rows").value,
        )

        now = time.monotonic()
        if est.offset_m is not None:
            self._last_lane_offset = float(np.clip(est.offset_m, -2.5, 2.5))
            self._last_lane_stamp = now
        elif now - self._last_lane_stamp > self.get_parameter("lane_hold_time_s").value:
            self._last_lane_offset = None

        if self._last_lane_offset is not None:
            self._lane_offset_pub.publish(Float32(data=self._last_lane_offset))

        self.get_logger().debug(
            f"[lane] offset={est.offset_m}  rows={est.rows_used}  width={est.lane_width_m}"
        )

        if self.get_parameter("publish_debug_image").value:
            self._publish_debug_image(rgb, mask, cam, est, msg.header)

    def _publish_debug_image(self, rgb, mask, cam: CameraModel, est, header):
        """Draw the road-line mask (green), used marking pixels (red/blue) and horizon."""
        debug = rgb.copy()
        debug[mask] = (0, 255, 0)
        for (v, u) in est.left_px:
            cv2.circle(debug, (u, v), 1, (255, 0, 0), -1)
        for (v, u) in est.right_px:
            cv2.circle(debug, (u, v), 1, (0, 0, 255), -1)
        h, w = debug.shape[:2]
        horizon = int(cam.cy - cam.fx * math.tan(cam.pitch_rad))
        cv2.line(debug, (0, horizon), (w, horizon), (255, 255, 0), 1)
        cv2.line(debug, (int(cam.cx), max(horizon, 0)), (int(cam.cx), h), (255, 255, 0), 1)
        text = "lane: none" if est.offset_m is None else f"lane offset {est.offset_m:+.2f} m"
        cv2.putText(debug, text, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        try:
            img_msg = self._cv_bridge.cv2_to_imgmsg(
                cv2.cvtColor(debug, cv2.COLOR_RGB2BGR), encoding="bgr8")
            img_msg.header = header
            self._debug_img_pub.publish(img_msg)
        except Exception as exc:
            self.get_logger().warn(f"Debug image publish failed: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()