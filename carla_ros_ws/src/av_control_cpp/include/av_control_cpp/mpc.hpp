// Linear time-varying lateral MPC on a rear-axle kinematic bicycle, ROS-free.
//
// Model (rear axle, path frame).  With e_y the lateral offset from the path
// (+ = left), e_psi the heading error and delta the front wheel angle:
//
//     e_y'   = v sin(e_psi)
//     e_psi' = v tan(delta) / L  -  v kappa(s)
//
// Linearized about the planned trajectory, i.e. about e = 0 and the
// feed-forward wheel angle delta_ref = atan(L kappa) that follows the path's
// curvature exactly, and discretized with step dt (forward Euler):
//
//     e_y[k+1]   = e_y[k] + dt v e_psi[k]
//     e_psi[k+1] = e_psi[k] + dt v sec^2(delta_ref_k) / L (delta_k - delta_ref_k)
//
// kappa_k is sampled where the car will be at step k (s0 + v k dt), so the
// model varies along the horizon (LTV).  Speed is held constant over the
// horizon at max(v, min_model_speed).
//
// QP over delta_0..delta_{N-1} (condensed: states eliminated):
//     min  sum_k  q_y e_y^2 + q_psi e_psi^2         (terminal x terminal_weight)
//              + r (delta_k - delta_ref_k)^2          control effort beyond feed-forward
//              + r_rate (delta_k - delta_{k-1})^2     control rate, delta_{-1} = last command
//     s.t. |delta_k| <= max_delta
//          |delta_k - delta_{k-1}| <= max_rate * dt
#pragma once

#include <string>
#include <vector>

#include "av_control_cpp/safety.hpp"

namespace av_control_cpp
{

struct MpcConfig
{
  int horizon{25};
  double dt{0.1};                 // s, model step
  double wheelbase{3.005};        // m, CARLA Model 3
  double max_delta{0.733};        // rad, wheel angle (0.6 x 70 deg, = Pure Pursuit's max_steer)
  double max_rate{0.8};           // rad/s, wheel angle rate
  double q_y{2.0};
  double q_psi{1.0};
  double r{0.5};
  double r_rate{10.0};
  double terminal_weight{5.0};
  double min_model_speed{1.0};    // m/s, keeps steering authority at a standstill
  int max_iter{4000};
  double eps{1e-5};
};

// Path-frame tracking errors and the curvature profile seen over the horizon.
struct Reference
{
  bool valid{false};
  double e_y{0.0};                // m, + = car left of the path
  double e_psi{0.0};              // rad, car heading - path heading
  double s0{0.0};                 // m, arc length of the projection
  std::vector<double> kappa;      // 1/m, at s0 + v k dt, k = 0..N-1
};

// Project the rear axle (rx, ry, yaw) onto the path polyline and sample its
// curvature (+ = left turn) every v*dt metres.  Beyond the path's end the
// curvature is 0.  Needs >= 2 path points.
Reference build_reference(const std::vector<Point> & path, double rx, double ry, double yaw,
                          double v, const MpcConfig & cfg);

// Prepend points extending the path `length` m backwards along a circle of
// curvature `kappa` tangent to its first point (straight if kappa = 0).
// The planner's path starts at or ahead of the vehicle centre, so the rear
// axle (where the MPC measures its errors) lies behind it.  Projecting onto
// the clamped first segment, or a straight extension, made the reference
// jump each time the car passed a vertex on a curve (steering-rate RMS
// 0.19 vs 0.04 rad/s).  Pass the path's *sustained* curvature
// (safety.hpp): on a curve the extension stays on the arc; a lane change's
// or swerve's S gives 0, because extrapolating its curvature backwards put
// 13 deg heading errors into the reference.
std::vector<Point> extend_back(const std::vector<Point> & path, double kappa,
                               double length = 6.0, double step = 1.5);

// Discrete model at one step: e[k+1] = A e[k] + B (delta - delta_ref) — A, B
// returned as (a12, b2): A = [[1, a12], [0, 1]], B = [0, b2].
struct StepModel
{
  double a12;
  double b2;
  double delta_ref;
};
StepModel linearize(double v, double kappa, const MpcConfig & cfg);

struct MpcResult
{
  bool ok{false};                 // solved (or solved inaccurate)
  std::string status;             // OSQP status string
  double delta{0.0};              // rad, first wheel angle (+ = left)
  std::vector<double> deltas;     // whole optimal sequence
  std::vector<double> e_y_pred;   // predicted e_y[1..N]
  int iterations{0};
  double solve_ms{0.0};
};

class LateralMpc
{
public:
  explicit LateralMpc(MpcConfig cfg = {});
  // v: current speed; prev_delta: wheel angle commanded last tick.
  MpcResult solve(const Reference & ref, double v, double prev_delta) const;
  const MpcConfig & config() const {return cfg_;}

private:
  MpcConfig cfg_;
};

// CARLA's normalized steer command for a wheel angle (+ delta = left):
// positive steer = RIGHT, normalized by the Model 3's ~70 deg max wheel angle.
double delta_to_carla_steer(double delta, double max_steer);

}  // namespace av_control_cpp
