// ROS-free longitudinal safety logic, ported from av_control/pure_pursuit.py
// so the MPC controller keeps every guard Pure Pursuit has: the obstacle
// corridor (path frame prediction + heading corridor), the speed governor
// (stop-and-hold, time-gap following), the end-of-path stop and the P-law.
//
// Everything is in the ROS map frame (x forward, y left, yaw CCW).
#pragma once

#include <deque>
#include <optional>
#include <utility>
#include <vector>

namespace av_control_cpp
{

struct Point
{
  double x{0.0};
  double y{0.0};
};

// An obstacle box with velocity (map frame).  Bare points use the default
// car size and zero velocity.
struct Obstacle
{
  double x{0.0}, y{0.0}, yaw{0.0};
  double half_length{0.0}, half_width{0.0};
  double vx{0.0}, vy{0.0};
};

// (gap to the box, obstacle speed along the path)
using Hit = std::pair<double, double>;

// Drop (in place) leading waypoints behind the vehicle.  Returns the last
// dropped waypoint, if any (the MPC keeps it to reference the rear axle).
std::optional<Point> prune_passed(std::vector<Point> & waypoints, double px, double py, double yaw);

double path_length_ahead(double px, double py, const std::vector<Point> & waypoints);

// Curvature common to the path's first `length` m (+ = left): net heading
// change / length if every resampled chord turns the same way, else 0.
double sustained_curvature(const std::vector<Point> & path, double length = 15.0,
                           double step = 2.5);

struct CorridorParams
{
  double corridor_half_width{1.35};
  double ego_half_length{2.4};
  double ego_speed{0.0};
  double max_predict_s{4.0};
  double min_speed{1.0};
  int predict_samples{8};
  double moving_threshold{0.5};
  double heading_check_m{0.0};
};

// Nearest obstacle blocking the corridor around the path (and the short
// corridor along the car's own heading).  See obstacle_along_path in
// pure_pursuit.py for the full rationale.
std::optional<Hit> obstacle_along_path(double px, double py, double yaw,
                                       const std::vector<Point> & waypoints,
                                       const std::vector<Obstacle> & obstacles,
                                       const CorridorParams & p);

// Speed that allows stopping `stop_distance` short of a point at `dist`.
double obstacle_speed_limit(std::optional<double> dist, double target_speed,
                            double stop_distance, double comfort_decel);

// (throttle, brake) from a P-law on speed error.
std::pair<double, double> speed_command(double speed, double speed_limit,
                                        double kp_throttle, double kp_brake);

// Median-filtered obstacle speed limit: stop-and-hold for parked obstacles,
// time-gap following (with hysteresis) for moving leads.
class ObstacleSpeedGovernor
{
public:
  ObstacleSpeedGovernor(double stop_distance, double comfort_decel, double resume_margin = 2.0,
                        int window = 5, double follow_time_gap = 1.0, double follow_gain = 0.5,
                        double moving_threshold = 0.5);

  double update(std::optional<double> obstacle_dist, double target_speed, double lead_speed = 0.0);
  std::optional<double> filtered_distance() const;
  bool holding() const {return holding_;}
  bool following() const {return following_;}

private:
  double stop_distance_, comfort_decel_, resume_margin_;
  std::size_t window_;
  double follow_time_gap_, follow_gain_, moving_threshold_;
  std::deque<double> hist_, speed_hist_;
  bool holding_{false};
  bool following_{false};
};

}  // namespace av_control_cpp
