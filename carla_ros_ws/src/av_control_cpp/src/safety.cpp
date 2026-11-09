// Port of the longitudinal safety logic in av_control/pure_pursuit.py.
// Keep the two in step: test/test_safety.cpp mirrors the Python tests.
#include "av_control_cpp/safety.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace av_control_cpp
{

namespace
{

constexpr double kInf = std::numeric_limits<double>::infinity();

double wrap(double a)
{
  return std::remainder(a, 2.0 * M_PI);   // (-pi, pi]
}

struct Closest
{
  double dist{kInf};
  double s{0.0};
  double ux{1.0}, uy{0.0};
  double lat{0.0};   // + = left
};

Closest closest_on_path(const std::vector<Point> & path, double ox, double oy)
{
  Closest best;
  double s_acc = 0.0;
  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    const double ax = path[i].x, ay = path[i].y;
    const double abx = path[i + 1].x - ax, aby = path[i + 1].y - ay;
    const double seg = std::hypot(abx, aby);
    if (seg < 1e-6) {
      continue;
    }
    const double t = std::clamp(((ox - ax) * abx + (oy - ay) * aby) / (seg * seg), 0.0, 1.0);
    const double d = std::hypot(ox - (ax + t * abx), oy - (ay + t * aby));
    if (d < best.dist) {
      best.dist = d;
      best.s = s_acc + t * seg;
      best.ux = abx / seg;
      best.uy = aby / seg;
      best.lat = -(ox - ax) * best.uy + (oy - ay) * best.ux;
    }
    s_acc += seg;
  }
  return best;
}

double median(std::deque<double> v)
{
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

std::optional<Point> prune_passed(std::vector<Point> & waypoints, double px, double py, double yaw)
{
  const double hx = std::cos(yaw), hy = std::sin(yaw);
  std::optional<Point> last;
  std::size_t n = 0;
  while (n < waypoints.size() &&
    (waypoints[n].x - px) * hx + (waypoints[n].y - py) * hy <= 0.0)
  {
    last = waypoints[n];
    ++n;
  }
  waypoints.erase(waypoints.begin(), waypoints.begin() + static_cast<long>(n));
  return last;
}

double path_length_ahead(double px, double py, const std::vector<Point> & waypoints)
{
  double total = 0.0;
  Point prev{px, py};
  for (const auto & w : waypoints) {
    total += std::hypot(w.x - prev.x, w.y - prev.y);
    prev = w;
  }
  return total;
}

double sustained_curvature(const std::vector<Point> & path, double length, double step)
{
  if (path.size() < 2) {
    return 0.0;
  }
  std::vector<double> s_acc{0.0};
  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    s_acc.push_back(s_acc.back() + std::hypot(path[i + 1].x - path[i].x, path[i + 1].y - path[i].y));
  }
  const double total = std::min(length, s_acc.back());
  const int n = static_cast<int>(std::floor(total / step));
  if (n < 3) {
    return 0.0;
  }
  std::vector<Point> pts;
  std::size_t k = 0;
  for (int i = 0; i <= n; ++i) {
    const double target = i * step;
    while (k + 2 < path.size() && s_acc[k + 1] < target) {
      ++k;
    }
    const double seg = std::max(s_acc[k + 1] - s_acc[k], 1e-9);
    const double f = std::clamp((target - s_acc[k]) / seg, 0.0, 1.0);
    pts.push_back({path[k].x + f * (path[k + 1].x - path[k].x),
        path[k].y + f * (path[k + 1].y - path[k].y)});
  }
  std::vector<double> head;
  for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
    head.push_back(std::atan2(pts[i + 1].y - pts[i].y, pts[i + 1].x - pts[i].x));
  }
  bool all_pos = true, all_neg = true;
  double sum = 0.0;
  std::size_t turns = 0;
  for (std::size_t i = 0; i + 1 < head.size(); ++i) {
    const double t = wrap(head[i + 1] - head[i]);
    all_pos = all_pos && t > 0.0;
    all_neg = all_neg && t < 0.0;
    sum += t;
    ++turns;
  }
  if (turns > 0 && (all_pos || all_neg)) {
    return sum / (step * static_cast<double>(turns));
  }
  return 0.0;
}

std::optional<Hit> obstacle_along_path(double px, double py, double yaw,
                                       const std::vector<Point> & waypoints,
                                       const std::vector<Obstacle> & obstacles,
                                       const CorridorParams & p)
{
  if (obstacles.empty()) {
    return std::nullopt;
  }
  std::vector<Point> path{{px, py}};
  path.insert(path.end(), waypoints.begin(), waypoints.end());
  if (path.size() < 2) {
    path.push_back({px + 50.0 * std::cos(yaw), py + 50.0 * std::sin(yaw)});
  }
  const double hx = std::cos(yaw), hy = std::sin(yaw);
  const double kappa = p.heading_check_m > 0.0 ? sustained_curvature(path) : 0.0;
  std::optional<Hit> best;
  auto consider = [&best](double gap, double v) {
      if (!best || gap < best->first) {
        best = Hit{gap, v};
      }
    };

  for (const auto & o : obstacles) {
    const double rvx = o.vx, rvy = o.vy;
    const bool moving = std::hypot(rvx, rvy) >= p.moving_threshold;
    const double vx = moving ? rvx : 0.0, vy = moving ? rvy : 0.0;
    const Closest c = closest_on_path(path, o.x, o.y);
    double s0 = c.s;
    if (s0 <= 0.0) {
      s0 = (o.x - px) * hx + (o.y - py) * hy;
    }
    const double v_al = vx * c.ux + vy * c.uy, v_lat = -vx * c.uy + vy * c.ux;
    const double rel = o.yaw - std::atan2(c.uy, c.ux);
    const double ext_along = o.half_length * std::abs(std::cos(rel)) +
      o.half_width * std::abs(std::sin(rel));
    const double ext_lat = o.half_length * std::abs(std::sin(rel)) +
      o.half_width * std::abs(std::cos(rel));
    const double t = std::min(p.max_predict_s, std::max(s0, 0.0) / std::max(p.ego_speed,
      p.min_speed));
    const int steps = moving ? p.predict_samples : 0;
    for (int k = 0; k <= steps; ++k) {
      const double tk = t * k / std::max(steps, 1);
      const double s_k = s0 + v_al * tk, lat_k = c.lat + v_lat * tk;
      if (s_k <= 0.0 || std::abs(lat_k) - ext_lat > p.corridor_half_width) {
        continue;
      }
      consider(std::max(0.0, s_k - ext_along - p.ego_half_length), rvx * c.ux + rvy * c.uy);
    }
    if (p.heading_check_m > 0.0) {
      const double a = (o.x - px) * hx + (o.y - py) * hy;
      const double lat = -(o.x - px) * hy + (o.y - py) * hx - 0.5 * kappa * a * a;
      const double r = o.yaw - (yaw + kappa * a);
      const double ea = o.half_length * std::abs(std::cos(r)) + o.half_width * std::abs(std::sin(r));
      const double el = o.half_length * std::abs(std::sin(r)) + o.half_width * std::abs(std::cos(r));
      if (a > 0.0 && a - ea <= p.ego_half_length + p.heading_check_m &&
        std::abs(lat) - el <= p.corridor_half_width)
      {
        consider(std::max(0.0, a - ea - p.ego_half_length), rvx * hx + rvy * hy);
      }
    }
  }
  return best;
}

double obstacle_speed_limit(std::optional<double> dist, double target_speed,
                            double stop_distance, double comfort_decel)
{
  if (!dist) {
    return target_speed;
  }
  const double free = *dist - stop_distance;
  if (free <= 0.0) {
    return 0.0;
  }
  return std::min(target_speed, std::sqrt(2.0 * comfort_decel * free));
}

std::pair<double, double> speed_command(double speed, double speed_limit,
                                        double kp_throttle, double kp_brake)
{
  if (speed_limit <= 0.05) {
    return {0.0, speed < 0.5 ? 1.0 : std::min(1.0, std::max(0.3, kp_brake * speed))};
  }
  const double err = speed_limit - speed;
  if (err >= 0.0) {
    return {std::min(1.0, kp_throttle * err), 0.0};
  }
  if (err > -0.5) {
    return {0.0, 0.0};   // coast
  }
  return {0.0, std::min(1.0, kp_brake * -err)};
}

ObstacleSpeedGovernor::ObstacleSpeedGovernor(double stop_distance, double comfort_decel,
                                             double resume_margin, int window,
                                             double follow_time_gap, double follow_gain,
                                             double moving_threshold)
: stop_distance_(stop_distance), comfort_decel_(comfort_decel), resume_margin_(resume_margin),
  window_(static_cast<std::size_t>(std::max(1, window))), follow_time_gap_(follow_time_gap),
  follow_gain_(follow_gain), moving_threshold_(moving_threshold)
{
}

double ObstacleSpeedGovernor::update(std::optional<double> obstacle_dist, double target_speed,
                                     double lead_speed)
{
  hist_.push_back(obstacle_dist ? *obstacle_dist : kInf);
  speed_hist_.push_back(obstacle_dist ? lead_speed : 0.0);
  while (hist_.size() > window_) {
    hist_.pop_front();
  }
  while (speed_hist_.size() > window_) {
    speed_hist_.pop_front();
  }
  const double filtered = median(hist_);
  const std::optional<double> dist =
    std::isinf(filtered) ? std::nullopt : std::optional<double>(filtered);
  const double v_lead = median(speed_hist_);

  const double lead_threshold = moving_threshold_ * (following_ ? 0.5 : 1.0);
  following_ = dist.has_value() && v_lead >= lead_threshold;
  if (following_) {
    holding_ = false;
    const double d_des = stop_distance_ + follow_time_gap_ * v_lead;
    const double v_follow = v_lead + follow_gain_ * (*dist - d_des);
    const double v_safe = obstacle_speed_limit(*dist + v_lead * v_lead / (2.0 * comfort_decel_),
        target_speed, stop_distance_, comfort_decel_);
    return std::max(0.0, std::min({target_speed, v_follow, v_safe}));
  }
  if (holding_) {
    if (*std::min_element(hist_.begin(), hist_.end()) <= stop_distance_ + resume_margin_) {
      return 0.0;
    }
    holding_ = false;
  }
  const double limit = obstacle_speed_limit(dist, target_speed, stop_distance_, comfort_decel_);
  if (limit <= 0.0) {
    holding_ = true;
  }
  return limit;
}

std::optional<double> ObstacleSpeedGovernor::filtered_distance() const
{
  if (hist_.empty()) {
    return std::nullopt;
  }
  const double f = median(hist_);
  return std::isinf(f) ? std::nullopt : std::optional<double>(f);
}

}  // namespace av_control_cpp
