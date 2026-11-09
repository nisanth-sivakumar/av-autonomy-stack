// mpc_controller_node: drop-in alternative to av_control's pure_pursuit_controller.
//
// Same topics:
//   sub  /carla/ego_vehicle/odometry      nav_msgs/Odometry
//        /planning/trajectory             nav_msgs/Path (latched)
//        /perception/objects              derived_object_msgs/ObjectArray
//        /perception/obstacles            geometry_msgs/PoseArray (fallback)
//   pub  /carla/ego_vehicle/vehicle_control_cmd   carla_msgs/CarlaEgoVehicleControl
//        /control/compute_time_ms         std_msgs/Float32 (MPC solve time)
//
// Lateral: LTV MPC (mpc.hpp) on the rear-axle kinematic bicycle.  If the
// solver fails, the steering is held and the car brakes.
// Longitudinal: the Pure Pursuit controller's safety logic, ported
// (safety.hpp): obstacle corridor + governor (stop-and-hold, time-gap
// following), end-of-path stop, no-perception hold, stale-trajectory
// brake and stuck recovery.  Parameter names match controller_node.py.
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "av_control_cpp/mpc.hpp"
#include "av_control_cpp/safety.hpp"
#include "carla_msgs/msg/carla_ego_vehicle_control.hpp"
#include "derived_object_msgs/msg/object_array.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"

namespace av_control_cpp
{

namespace
{
constexpr double kDefaultHalfLength = 2.2;
constexpr double kDefaultHalfWidth = 0.9;

double yaw_of(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}
}  // namespace

class MpcController : public rclcpp::Node
{
public:
  MpcController()
  : Node("mpc_controller")
  {
    // shared with controller_node.py
    target_speed_ = declare_parameter("target_speed", 3.0);
    kp_speed_ = declare_parameter("kp_speed", 0.5);
    kp_brake_ = declare_parameter("kp_brake", 0.5);
    max_steer_ = declare_parameter("max_steer", 0.6);
    ego_half_length_ = declare_parameter("ego_half_length", 2.4);
    debug_interval_ = declare_parameter("debug_interval", 10);
    use_obstacles_ = declare_parameter("use_obstacles", true);
    require_perception_ = declare_parameter("require_perception", true);
    corridor_.corridor_half_width = declare_parameter("corridor_half_width_m", 1.35);
    stop_distance_ = declare_parameter("stop_distance_m", 3.5);
    comfort_decel_ = declare_parameter("comfort_decel", 2.0);
    const double resume_margin = declare_parameter("resume_margin_m", 2.0);
    const int window = declare_parameter("obstacle_filter_window", 5);
    obstacle_timeout_ = declare_parameter("obstacle_timeout", 1.0);
    corridor_.max_predict_s = declare_parameter("obstacle_predict_s", 4.0);
    corridor_.heading_check_m = declare_parameter("heading_check_m", 4.5);
    const double follow_time_gap = declare_parameter("follow_time_gap_s", 1.0);
    const double follow_gain = declare_parameter("follow_gain", 0.5);
    end_stop_margin_ = declare_parameter("end_stop_margin_m", 1.0);
    stuck_speed_ = declare_parameter("stuck_speed_thresh", 0.2);
    stuck_timeout_ = declare_parameter("stuck_timeout", 3.0);
    stuck_recovery_ = declare_parameter("stuck_recovery_time", 2.0);
    stuck_max_retries_ = declare_parameter("stuck_max_retries", 3);
    trajectory_timeout_ = declare_parameter("trajectory_timeout", 2.0);
    corridor_.ego_half_length = ego_half_length_;

    // MPC
    MpcConfig cfg;
    cfg.horizon = declare_parameter("mpc_horizon", cfg.horizon);
    cfg.dt = declare_parameter("mpc_dt", cfg.dt);
    cfg.wheelbase = declare_parameter("wheelbase", cfg.wheelbase);
    cfg.max_delta = max_steer_ * 70.0 * M_PI / 180.0;
    cfg.max_rate = declare_parameter("mpc_max_steer_rate", cfg.max_rate);
    cfg.q_y = declare_parameter("mpc_q_lateral", cfg.q_y);
    cfg.q_psi = declare_parameter("mpc_q_heading", cfg.q_psi);
    cfg.r = declare_parameter("mpc_r_steer", cfg.r);
    cfg.r_rate = declare_parameter("mpc_r_steer_rate", cfg.r_rate);
    cfg.terminal_weight = declare_parameter("mpc_terminal_weight", cfg.terminal_weight);
    rear_axle_offset_ = declare_parameter("rear_axle_offset_m", 1.5);
    mpc_ = std::make_unique<LateralMpc>(cfg);

    governor_ = std::make_unique<ObstacleSpeedGovernor>(
      stop_distance_, comfort_decel_, resume_margin, window, follow_time_gap, follow_gain);

    auto latched = rclcpp::QoS(1).reliable().transient_local();
    cmd_pub_ = create_publisher<carla_msgs::msg::CarlaEgoVehicleControl>(
      "/carla/ego_vehicle/vehicle_control_cmd", 10);
    time_pub_ = create_publisher<std_msgs::msg::Float32>("/control/compute_time_ms", 10);
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/carla/ego_vehicle/odometry", rclcpp::QoS(10).reliable(),
      [this](nav_msgs::msg::Odometry::SharedPtr m) {on_odom(*m);});
    traj_sub_ = create_subscription<nav_msgs::msg::Path>(
      "/planning/trajectory", latched, [this](nav_msgs::msg::Path::SharedPtr m) {
        waypoints_.clear();
        for (const auto & ps : m->poses) {
          waypoints_.push_back({ps.pose.position.x, ps.pose.position.y});
        }
        traj_stamp_ = now_s();
      });
    obj_sub_ = create_subscription<derived_object_msgs::msg::ObjectArray>(
      "/perception/objects", 10, [this](derived_object_msgs::msg::ObjectArray::SharedPtr m) {
        objects_.clear();
        for (const auto & o : m->objects) {
          const auto & d = o.shape.dimensions;
          objects_.push_back({o.pose.position.x, o.pose.position.y, yaw_of(o.pose.orientation),
              d.size() > 0 && d[0] > 0 ? d[0] / 2.0 : kDefaultHalfLength,
              d.size() > 1 && d[1] > 0 ? d[1] / 2.0 : kDefaultHalfWidth,
              o.twist.linear.x, o.twist.linear.y});
        }
        objects_stamp_ = now_s();
      });
    pts_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      "/perception/obstacles", 10, [this](geometry_msgs::msg::PoseArray::SharedPtr m) {
        points_.clear();
        for (const auto & p : m->poses) {
          points_.push_back({p.position.x, p.position.y, 0.0, kDefaultHalfLength,
              kDefaultHalfWidth, 0.0, 0.0});
        }
        points_stamp_ = now_s();
      });
    RCLCPP_INFO(get_logger(), "MpcController ready (horizon %d x %.2f s, wheelbase %.3f m) — "
      "waiting for /planning/trajectory and odometry …", cfg.horizon, cfg.dt, cfg.wheelbase);
  }

private:
  double now_s() {return get_clock()->now().seconds();}

  void publish(double throttle, double steer, double brake)
  {
    carla_msgs::msg::CarlaEgoVehicleControl cmd;
    cmd.throttle = static_cast<float>(throttle);
    cmd.steer = static_cast<float>(steer);
    cmd.brake = static_cast<float>(brake);
    cmd.hand_brake = false;
    cmd.reverse = false;
    cmd.manual_gear_shift = false;
    cmd_pub_->publish(cmd);
  }

  // Latest obstacle list, or nullopt if perception is absent or stale.
  std::optional<std::vector<Obstacle>> current_obstacles(double now) const
  {
    if (objects_stamp_ && now - *objects_stamp_ <= obstacle_timeout_) {
      return objects_;
    }
    if (points_stamp_ && now - *points_stamp_ <= obstacle_timeout_) {
      return points_;
    }
    return std::nullopt;
  }

  bool check_stuck(double throttle, double speed, double now)
  {
    if (speed >= stuck_speed_) {
      pushing_since_.reset();
      stuck_retries_ = 0;
      return false;
    }
    if (throttle <= 0.05) {
      pushing_since_.reset();
      return false;
    }
    if (!pushing_since_) {
      pushing_since_ = now;
      return false;
    }
    if (now - *pushing_since_ <= stuck_timeout_) {
      return false;
    }
    if (stuck_retries_ >= stuck_max_retries_) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
        "STUCK: retries exhausted — holding brake. Reset the vehicle and relaunch.");
      return true;
    }
    ++stuck_retries_;
    recover_until_ = now + stuck_recovery_;
    RCLCPP_WARN(get_logger(), "STUCK: speed=%.2f m/s with throttle applied for >%.0f s — "
      "braking %.0f s then retrying.", speed, stuck_timeout_, stuck_recovery_);
    return true;
  }

  void on_odom(const nav_msgs::msg::Odometry & msg)
  {
    const double px = msg.pose.pose.position.x, py = msg.pose.pose.position.y;
    const double yaw = yaw_of(msg.pose.pose.orientation);
    const double speed = std::hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y);
    if (std::abs(px) < 0.1 && std::abs(py) < 0.1) {
      return;   // CARLA startup artefact
    }
    const bool debug = (++tick_ % debug_interval_) == 0;
    const double now = now_s();

    if (!traj_stamp_) {
      publish(0.0, 0.0, 0.0);
      return;
    }
    if (now - *traj_stamp_ > trajectory_timeout_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Trajectory stale (%.1f s) — braking.", now - *traj_stamp_);
      publish(0.0, 0.0, 1.0);
      return;
    }
    if (recover_until_) {
      if (now < *recover_until_) {
        publish(0.0, 0.0, 1.0);
        return;
      }
      recover_until_.reset();
      pushing_since_.reset();
      RCLCPP_INFO(get_logger(), "Stuck recovery: retrying (%d/%d).", stuck_retries_,
        stuck_max_retries_);
    }

    prune_passed(waypoints_, px, py, yaw);
    if (waypoints_.empty()) {
      publish(0.0, 0.0, speed > 0.5 ? 0.3 : 1.0);   // no path ahead: hold still
      return;
    }

    // ── Lateral: MPC about the rear axle ──
    const double rx = px - rear_axle_offset_ * std::cos(yaw);
    const double ry = py - rear_axle_offset_ * std::sin(yaw);
    std::vector<Point> path(waypoints_);
    if (path.size() < 2) {
      path.insert(path.begin(), Point{px, py});
    }
    // cover the rear axle: extend back along the road's sustained curvature
    path = extend_back(path, sustained_curvature(path), rear_axle_offset_ + 4.5);
    const Reference ref = build_reference(path, rx, ry, yaw, speed, mpc_->config());
    const MpcResult res = mpc_->solve(ref, speed, prev_delta_);
    std_msgs::msg::Float32 tmsg;
    tmsg.data = static_cast<float>(res.solve_ms);
    time_pub_->publish(tmsg);
    bool solver_failed = !res.ok;
    if (res.ok) {
      prev_delta_ = res.delta;
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
        "MPC failed (%s) — holding steering, braking.", res.status.c_str());
    }
    const double steer = delta_to_carla_steer(prev_delta_, max_steer_);

    // ── Longitudinal: Pure Pursuit's safety logic ──
    std::optional<Hit> hit;
    bool perception_ok = true;
    if (use_obstacles_) {
      const auto obstacles = current_obstacles(now);
      perception_ok = obstacles.has_value() || !require_perception_;
      CorridorParams cp = corridor_;
      cp.ego_speed = speed;
      hit = obstacle_along_path(px, py, yaw, waypoints_, obstacles.value_or(
          std::vector<Obstacle>{}), cp);
    }
    const double obstacle_limit = governor_->update(
      hit ? std::optional<double>(hit->first) : std::nullopt, target_speed_,
      hit ? hit->second : 0.0);
    const double end_limit = obstacle_speed_limit(path_length_ahead(px, py, waypoints_),
        target_speed_, end_stop_margin_, comfort_decel_);
    double speed_limit = std::min(obstacle_limit, end_limit);
    if (!perception_ok) {
      speed_limit = 0.0;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "No fresh /perception/objects — holding.");
    }
    if (solver_failed) {
      speed_limit = 0.0;
    }
    auto [throttle, brake] = speed_command(speed, speed_limit, kp_speed_, kp_brake_);
    if (check_stuck(throttle, speed, now)) {
      publish(0.0, 0.0, 1.0);
      return;
    }
    if (debug) {
      const auto g = governor_->filtered_distance();
      RCLCPP_INFO(get_logger(),
        "[ctrl] pos=(%.1f,%.1f) yaw=%.0f° v=%.2f/%.2f m/s  steer=%+.3f  thr=%.2f brk=%.2f  "
        "obstacle=%s  e_y=%+.2f e_psi=%+.1f°  mpc=%s %.2f ms (%d it)  wps=%zu",
        px, py, yaw * 180.0 / M_PI, speed, speed_limit, steer, throttle, brake,
        g ? (std::to_string(*g).substr(0, 4) + " m").c_str() : "clear",
        ref.e_y, ref.e_psi * 180.0 / M_PI, res.status.c_str(), res.solve_ms, res.iterations,
        waypoints_.size());
    }
    publish(throttle, steer, brake);
  }

  // parameters
  double target_speed_, kp_speed_, kp_brake_, max_steer_, ego_half_length_;
  int64_t debug_interval_;
  bool use_obstacles_, require_perception_;
  double stop_distance_, comfort_decel_, obstacle_timeout_, end_stop_margin_;
  double stuck_speed_, stuck_timeout_, stuck_recovery_;
  int stuck_max_retries_;
  double trajectory_timeout_, rear_axle_offset_;
  CorridorParams corridor_;

  std::unique_ptr<LateralMpc> mpc_;
  std::unique_ptr<ObstacleSpeedGovernor> governor_;

  // state
  std::vector<Point> waypoints_;
  std::vector<Obstacle> objects_, points_;
  std::optional<double> traj_stamp_, objects_stamp_, points_stamp_;
  std::optional<double> pushing_since_, recover_until_;
  int stuck_retries_{0};
  int64_t tick_{0};
  double prev_delta_{0.0};

  rclcpp::Publisher<carla_msgs::msg::CarlaEgoVehicleControl>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr time_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr traj_sub_;
  rclcpp::Subscription<derived_object_msgs::msg::ObjectArray>::SharedPtr obj_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr pts_sub_;
};

}  // namespace av_control_cpp

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<av_control_cpp::MpcController>());
  rclcpp::shutdown();
  return 0;
}
