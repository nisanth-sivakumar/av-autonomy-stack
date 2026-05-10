#!/usr/bin/env python3
"""
perception_node.py — av_perception package  (Phase 3)
======================================================
Provides two perception capabilities:

  1. Object detection
     Queries the CARLA Python API for nearby vehicles and pedestrians,
     adds Gaussian noise to simulate real sensor uncertainty, and
     publishes obstacle positions in the ROS map frame.

  2. Lane detection
     Processes the semantic segmentation camera image to locate lane
     markings, computes the lateral offset of the ego vehicle from the
     lane centre, and publishes it for downstream use.

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
  /carla/ego_vehicle/semantic_segmentation_front/image  (sensor_msgs/Image)
  /carla/ego_vehicle/odometry                           (nav_msgs/Odometry)

Connects to CARLA Python API (read-only) to query actor positions.

Coordinate conventions
----------------------
    ros_x =  carla_x
    ros_y = -carla_y
    Obstacle positions published in ROS map frame.

CARLA semantic labels used
--------------------------
    6 = RoadLine  (lane markings — bright green-yellow in colour palette)
    7 = Road
"""

import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from sensor_msgs.msg import Image
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


# ---------------------------------------------------------------------------
# Semantic segmentation label constants (CARLA 0.9.x)
# ---------------------------------------------------------------------------
LABEL_ROAD_LINE = 6
LABEL_ROAD      = 7
LABEL_VEHICLE   = 10
LABEL_PEDESTRIAN = 4


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

        # Lane detection
        self.declare_parameter("lane_roi_top_frac",     0.55)   # ROI starts at 55% down image
        self.declare_parameter("pixels_per_meter",      80.0)   # rough calibration — tune per camera
        self.declare_parameter("min_lane_pixels",       20)     # min pixels to trust detection
        self.declare_parameter("publish_debug_image",   True)

        # ── State ─────────────────────────────────────────────────────────
        self._ego_x:     float | None = None
        self._ego_y:     float | None = None
        self._ego_yaw:   float | None = None

        self._carla_world    = None
        self._ego_actor_id:  int | None = None
        self._obstacles:     list[tuple[float, float]] = []

        self._cv_bridge = CvBridge() if CV_AVAILABLE else None
        self._last_lane_offset: float = 0.0

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

    def _semantic_image_cb(self, msg: Image):
        """
        Process the semantic segmentation camera image to detect lane markings
        and compute the lateral offset of the ego vehicle from lane centre.

        Method
        ------
        1. Convert ROS Image → NumPy array via cv_bridge.
        2. Extract the R channel, which in CARLA's semantic seg output contains
           the semantic label ID (0-22).
        3. Create a road-line mask (label == 6) within the bottom ROI.
        4. Find left and right lane marking pixel columns.
        5. Compute lane centre and offset from image centre.
        6. Convert pixel offset → metres using pixels_per_meter calibration.
        7. Publish lane offset and annotated debug image.

        Coordinate sign convention
        --------------------------
        Positive lane_offset = ego is LEFT of lane centre → planner should
        bias trajectory slightly right.
        """
        if not CV_AVAILABLE:
            return

        try:
            # Convert to BGR numpy array, then to RGB
            bgr = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return

        h, w = rgb.shape[:2]
        roi_top     = int(h * self.get_parameter("lane_roi_top_frac").value)
        min_pixels  = self.get_parameter("min_lane_pixels").value
        ppm         = self.get_parameter("pixels_per_meter").value

        # ── Extract semantic labels ───────────────────────────────────────
        # CARLA semantic seg: R channel = semantic label ID.
        # This works with both raw label output and carla_ros_bridge encoding.
        labels = rgb[:, :, 0]   # R channel contains the semantic label

        # ROI: bottom portion of image (close to vehicle, most reliable)
        roi_labels  = labels[roi_top:, :]
        roi_rgb     = rgb[roi_top:, :].copy()

        # ── Road line mask ────────────────────────────────────────────────
        lane_mask = (roi_labels == LABEL_ROAD_LINE).astype(np.uint8) * 255

        # Fallback: if label approach finds nothing, try colour matching.
        # CARLA colours road lines as (157, 234, 50) in its palette.
        if cv2.countNonZero(lane_mask) < min_pixels:
            lower = np.array([140, 220, 35], dtype=np.uint8)
            upper = np.array([175, 245, 65], dtype=np.uint8)
            lane_mask = cv2.inRange(roi_rgb, lower, upper)

        # ── Compute lane offset ───────────────────────────────────────────
        center_x    = w // 2
        lane_offset = None

        if cv2.countNonZero(lane_mask) >= min_pixels:
            ys, xs = np.where(lane_mask > 0)
            left_xs  = xs[xs < center_x]
            right_xs = xs[xs > center_x]

            if len(left_xs) >= min_pixels // 2 and len(right_xs) >= min_pixels // 2:
                # Both lane markings visible — use midpoint
                left_mean  = float(np.mean(left_xs))
                right_mean = float(np.mean(right_xs))
                lane_center_x = (left_mean + right_mean) / 2.0
                offset_px   = center_x - lane_center_x   # + = ego left of centre
                lane_offset = offset_px / ppm

            elif len(left_xs) >= min_pixels // 2:
                # Only left lane visible — estimate from left marking
                # Assume standard lane width ~3.75 m, so right marking is ~1.875m right
                left_mean   = float(np.mean(left_xs))
                half_lane_px = 1.875 * ppm
                lane_center_x = left_mean + half_lane_px
                offset_px   = center_x - lane_center_x
                lane_offset = offset_px / ppm

            elif len(right_xs) >= min_pixels // 2:
                # Only right lane visible — estimate from right marking
                right_mean  = float(np.mean(right_xs))
                half_lane_px = 1.875 * ppm
                lane_center_x = right_mean - half_lane_px
                offset_px   = center_x - lane_center_x
                lane_offset = offset_px / ppm

        # Clamp and publish
        if lane_offset is not None:
            lane_offset = float(np.clip(lane_offset, -2.5, 2.5))
            self._last_lane_offset = lane_offset
        # else: keep last known offset (handles momentary detection failures)

        offset_msg = Float32()
        offset_msg.data = self._last_lane_offset
        self._lane_offset_pub.publish(offset_msg)

        self.get_logger().debug(
            f"[lane] offset={self._last_lane_offset:+.3f} m  "
            f"lane_pixels={cv2.countNonZero(lane_mask)}"
        )

        # ── Debug image ───────────────────────────────────────────────────
        if self.get_parameter("publish_debug_image").value:
            self._publish_debug_image(rgb, roi_top, lane_mask, lane_offset)

    def _publish_debug_image(
        self,
        rgb:        np.ndarray,
        roi_top:    int,
        lane_mask:  np.ndarray,
        lane_offset: float | None,
    ):
        """Overlay lane mask and centre-line marker on image for debugging."""
        debug = rgb.copy()
        h, w  = debug.shape[:2]

        # Overlay lane mask in green on the ROI
        lane_overlay       = debug[roi_top:].copy()
        mask_bool          = lane_mask > 0
        lane_overlay[mask_bool, 0] = 0
        lane_overlay[mask_bool, 1] = 255
        lane_overlay[mask_bool, 2] = 0
        debug[roi_top:]    = cv2.addWeighted(debug[roi_top:], 0.6, lane_overlay, 0.4, 0)

        # Draw image centre line
        cv2.line(debug, (w // 2, roi_top), (w // 2, h), (255, 255, 0), 1)

        # Draw computed lane centre
        if lane_offset is not None:
            ppm = self.get_parameter("pixels_per_meter").value
            lane_cx = int(w // 2 - lane_offset * ppm)
            cv2.line(debug, (lane_cx, roi_top), (lane_cx, h), (0, 255, 255), 2)

        # Annotate offset value
        text  = f"Lane offset: {self._last_lane_offset:+.2f} m"
        color = (0, 200, 0) if abs(self._last_lane_offset) < 0.5 else (0, 0, 255)
        cv2.putText(debug, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Convert back to ROS Image and publish
        try:
            debug_bgr = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)
            img_msg   = self._cv_bridge.cv2_to_imgmsg(debug_bgr, encoding="bgr8")
            img_msg.header.stamp    = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "ego_vehicle"
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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()