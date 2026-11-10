#include "av_control_cpp/mpc.hpp"

#include "osqp.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>

namespace av_control_cpp
{

namespace
{

double wrap(double a)
{
  return std::remainder(a, 2.0 * M_PI);
}

// Dense matrix -> OSQP CSC.  Keeps entries with |v| > 0 (upper triangle only
// when `upper` is set, as OSQP needs for P).
struct Csc
{
  std::vector<c_float> x;
  std::vector<c_int> i, p;
  csc * m{nullptr};

  Csc(const Eigen::MatrixXd & d, bool upper)
  {
    const auto rows = static_cast<c_int>(d.rows()), cols = static_cast<c_int>(d.cols());
    p.push_back(0);
    for (c_int c = 0; c < cols; ++c) {
      for (c_int r = 0; r < rows; ++r) {
        if (upper && r > c) {
          break;
        }
        const double v = d(r, c);
        if (v != 0.0) {
          x.push_back(v);
          i.push_back(r);
        }
      }
      p.push_back(static_cast<c_int>(x.size()));
    }
    m = csc_matrix(rows, cols, static_cast<c_int>(x.size()), x.data(), i.data(), p.data());
  }
  ~Csc() {c_free(m);}   // frees the struct only; arrays are owned by the vectors
  Csc(const Csc &) = delete;
  Csc & operator=(const Csc &) = delete;
};

}  // namespace

Reference build_reference(const std::vector<Point> & path, double rx, double ry, double yaw,
                          double v, const MpcConfig & cfg)
{
  Reference ref;
  if (path.size() < 2) {
    return ref;
  }
  // segment headings, lengths and cumulative arc length
  const std::size_t m = path.size() - 1;
  std::vector<double> head(m), len(m), s_start(m + 1, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    const double dx = path[i + 1].x - path[i].x, dy = path[i + 1].y - path[i].y;
    head[i] = std::atan2(dy, dx);
    len[i] = std::hypot(dx, dy);
    s_start[i + 1] = s_start[i] + len[i];
  }
  // closest point (segments shorter than 1 cm are skipped)
  double best = std::numeric_limits<double>::infinity();
  std::size_t seg = 0;
  double seg_t = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    if (len[i] < 1e-2) {
      continue;
    }
    const double ux = std::cos(head[i]), uy = std::sin(head[i]);
    const double t = std::clamp((rx - path[i].x) * ux + (ry - path[i].y) * uy, 0.0, len[i]);
    const double d = std::hypot(rx - (path[i].x + t * ux), ry - (path[i].y + t * uy));
    if (d < best) {
      best = d;
      seg = i;
      seg_t = t;
      ref.s0 = s_start[i] + t;
    }
  }
  if (!std::isfinite(best)) {
    return ref;
  }
  // curvature at each vertex: heading change over the mean adjacent length
  std::vector<double> kv(m + 1, 0.0);
  for (std::size_t i = 1; i < m; ++i) {
    const double l = 0.5 * (len[i - 1] + len[i]);
    if (l > 1e-2) {
      kv[i] = wrap(head[i] - head[i - 1]) / l;
    }
  }
  // End vertices have no turn of their own: take the neighbour's curvature.
  // Leaving them at 0 pulled the feed-forward toward a straight whenever the
  // rear axle was on the first segment, and as waypoints shifted through
  // the window every 3 m that stepped the steering (rate RMS 0.18 vs 0.04).
  if (m >= 2) {
    kv[0] = kv[1];
    kv[m] = kv[m - 1];
  }
  const double ux = std::cos(head[seg]), uy = std::sin(head[seg]);
  // Lateral error from the chord, corrected to the arc through the vertices:
  // the arc bulges kappa f(1-f) l^2 / 2 outward of the chord (up to 3.75 cm
  // for 3 m chords on R 30 m), so a car tracking the arc saw e_y oscillate
  // once per chord and the MPC chased it.
  {
    const double fl = len[seg] > 1e-9 ? seg_t / len[seg] : 0.0;
    const double k_here = kv[seg] + fl * (kv[seg + 1] - kv[seg]);
    const double bulge = 0.5 * k_here * fl * (1.0 - fl) * len[seg] * len[seg];
    ref.e_y = -(rx - path[seg].x) * uy + (ry - path[seg].y) * ux + bulge;
  }
  // Path heading interpolated along the segment between vertex headings (the
  // mean of the adjacent chords).  A chord's own heading jumps at every
  // vertex (5.7 deg per 3 m on R 30 m), which made e_psi, and the steering,
  // step: steering-rate RMS 2.5x Pure Pursuit's on an empty curve.
  auto vertex_heading = [&](std::size_t v) {
      if (v == 0) {
        return head[0];
      }
      if (v >= m) {
        return head[m - 1];
      }
      return head[v - 1] + 0.5 * wrap(head[v] - head[v - 1]);
    };
  const double f = len[seg] > 1e-9 ? seg_t / len[seg] : 0.0;
  const double h0 = vertex_heading(seg), h1 = vertex_heading(seg + 1);
  ref.e_psi = wrap(yaw - (h0 + f * wrap(h1 - h0)));

  const double vm = std::max(v, cfg.min_model_speed);
  ref.kappa.resize(static_cast<std::size_t>(cfg.horizon), 0.0);
  std::size_t j = 0;
  for (int k = 0; k < cfg.horizon; ++k) {
    const double s = ref.s0 + vm * cfg.dt * k;
    if (s >= s_start[m]) {
      break;                       // beyond the end: straight
    }
    while (j + 1 < m && s_start[j + 1] <= s) {
      ++j;
    }
    // linear between the segment's end vertices: a step per vertex made the
    // feed-forward wheel angle step every 3 m (steering-rate RMS 0.19 vs
    // 0.07 rad/s with 1 m waypoints in a closed-loop experiment)
    const double f = len[j] > 1e-9 ? std::clamp((s - s_start[j]) / len[j], 0.0, 1.0) : 0.0;
    ref.kappa[static_cast<std::size_t>(k)] = kv[j] + f * (kv[j + 1] - kv[j]);
  }
  ref.valid = true;
  return ref;
}

std::vector<Point> extend_back(const std::vector<Point> & path, double kappa, double length,
                               double step)
{
  if (path.size() < 2) {
    return path;
  }
  // tangent at the first point: the first chord's heading is the arc's at
  // the chord's middle
  const double l0 = std::hypot(path[1].x - path[0].x, path[1].y - path[0].y);
  const double hs = std::atan2(path[1].y - path[0].y, path[1].x - path[0].x) - 0.5 * kappa * l0;
  std::vector<Point> out;
  const int n = static_cast<int>(std::ceil(length / step));
  for (int i = n; i >= 1; --i) {
    const double d = -i * step;        // arc length from path[0], negative = behind
    double x, y;
    if (std::abs(kappa) < 1e-6) {
      x = path[0].x + d * std::cos(hs);
      y = path[0].y + d * std::sin(hs);
    } else {
      x = path[0].x + (std::sin(hs + kappa * d) - std::sin(hs)) / kappa;
      y = path[0].y - (std::cos(hs + kappa * d) - std::cos(hs)) / kappa;
    }
    out.push_back({x, y});
  }
  out.insert(out.end(), path.begin(), path.end());
  return out;
}

std::vector<Point> back_from_previous(const std::vector<Point> & prev,
                                      const std::vector<Point> & next, double length,
                                      double max_offset, double max_rotation)
{
  std::vector<Point> out;
  if (prev.size() < 2 || next.size() < 2) {
    return out;
  }
  const Point n0 = next[0];
  // projection of next[0] onto prev (closest point over all segments)
  double best = std::numeric_limits<double>::infinity();
  std::size_t seg = 0;
  double seg_t = 0.0;
  for (std::size_t i = 0; i + 1 < prev.size(); ++i) {
    const double dx = prev[i + 1].x - prev[i].x, dy = prev[i + 1].y - prev[i].y;
    const double l = std::hypot(dx, dy);
    if (l < 1e-2) {
      continue;
    }
    const double t = std::clamp(((n0.x - prev[i].x) * dx + (n0.y - prev[i].y) * dy) / l, 0.0, l);
    const double d = std::hypot(n0.x - (prev[i].x + t * dx / l), n0.y - (prev[i].y + t * dy / l));
    if (d < best - 1e-9) {
      best = d;
      seg = i;
      seg_t = t;
    }
  }
  if (!std::isfinite(best) || best > max_offset) {
    return out;
  }
  const double sdx = prev[seg + 1].x - prev[seg].x, sdy = prev[seg + 1].y - prev[seg].y;
  const double sl = std::hypot(sdx, sdy);
  // a projection at a segment's very end belongs to the next segment: compare
  // like chords (the one leaving the point), or a vertex's own turn reads as
  // a rotation
  std::size_t hseg = seg;
  if (seg_t > sl - 1e-3 && seg + 2 < prev.size()) {
    hseg = seg + 1;
  }
  const double h_prev = std::atan2(prev[hseg + 1].y - prev[hseg].y, prev[hseg + 1].x - prev[hseg].x);
  const double h_next = std::atan2(next[1].y - next[0].y, next[1].x - next[0].x);
  const double rot = std::remainder(h_next - h_prev, 2.0 * M_PI);
  if (std::abs(rot) > max_rotation) {
    return out;
  }
  const Point p{prev[seg].x + seg_t * sdx / sl, prev[seg].y + seg_t * sdy / sl};
  const double c = std::cos(rot), s = std::sin(rot);
  auto place = [&](const Point & q) {
      const double dx = q.x - p.x, dy = q.y - p.y;
      return Point{n0.x + c * dx - s * dy, n0.y + s * dx + c * dy};
    };
  // walk prev backwards from the projection
  double acc = seg_t;
  std::vector<Point> rev;
  if (seg_t > 1e-3) {
    rev.push_back(place(prev[seg]));
  }
  for (std::size_t i = seg; i > 0 && acc < length; --i) {
    acc += std::hypot(prev[i].x - prev[i - 1].x, prev[i].y - prev[i - 1].y);
    rev.push_back(place(prev[i - 1]));
  }
  out.assign(rev.rbegin(), rev.rend());
  return out;
}

StepModel linearize(double v, double kappa, const MpcConfig & cfg)
{
  const double vm = std::max(v, cfg.min_model_speed);
  const double delta_ref = std::atan(cfg.wheelbase * kappa);
  const double sec2 = 1.0 + std::pow(std::tan(delta_ref), 2);
  return {cfg.dt * vm, cfg.dt * vm * sec2 / cfg.wheelbase, delta_ref};
}

LateralMpc::LateralMpc(MpcConfig cfg)
: cfg_(cfg) {}

MpcResult LateralMpc::solve(const Reference & ref, double v, double prev_delta) const
{
  const auto t0 = std::chrono::steady_clock::now();
  MpcResult res;
  const int n = cfg_.horizon;
  if (!ref.valid || static_cast<int>(ref.kappa.size()) != n) {
    res.status = "invalid reference";
    return res;
  }

  // ── Condensed prediction: E = Sx e0 + Su delta + Sc, E = [e_1; ...; e_N] ──
  Eigen::MatrixXd Su = Eigen::MatrixXd::Zero(2 * n, n);
  Eigen::VectorXd free_resp(2 * n);     // Sx e0 + Sc
  Eigen::VectorXd dref(n);
  Eigen::Vector2d e(ref.e_y, ref.e_psi);
  std::vector<StepModel> steps;
  for (int k = 0; k < n; ++k) {
    steps.push_back(linearize(v, ref.kappa[static_cast<std::size_t>(k)], cfg_));
    dref(k) = steps.back().delta_ref;
  }
  // free response (delta = delta_ref)
  for (int k = 0; k < n; ++k) {
    e = Eigen::Vector2d(e(0) + steps[k].a12 * e(1), e(1));
    free_resp.segment<2>(2 * k) = e;
  }
  // forced response: column j = effect of a unit (delta_j - delta_ref_j)
  for (int j = 0; j < n; ++j) {
    Eigen::Vector2d x(0.0, steps[j].b2);
    for (int k = j; k < n; ++k) {
      if (k > j) {
        x = Eigen::Vector2d(x(0) + steps[k].a12 * x(1), x(1));
      }
      Su.block<2, 1>(2 * k, j) = x;
    }
  }
  // E = free_resp + Su (delta - dref)  →  E = Su delta + (free_resp - Su dref)
  const Eigen::VectorXd offset = free_resp - Su * dref;

  Eigen::VectorXd qdiag(2 * n);
  for (int k = 0; k < n; ++k) {
    const double w = (k == n - 1) ? cfg_.terminal_weight : 1.0;
    qdiag(2 * k) = cfg_.q_y * w;
    qdiag(2 * k + 1) = cfg_.q_psi * w;
  }
  // difference operator: (D delta)_k = delta_k - delta_{k-1}, delta_{-1} = prev
  Eigen::MatrixXd D = Eigen::MatrixXd::Identity(n, n);
  for (int k = 1; k < n; ++k) {
    D(k, k - 1) = -1.0;
  }
  Eigen::VectorXd d0 = Eigen::VectorXd::Zero(n);
  d0(0) = prev_delta;

  const Eigen::MatrixXd Q = qdiag.asDiagonal();
  Eigen::MatrixXd P = Su.transpose() * Q * Su +
    cfg_.r * Eigen::MatrixXd::Identity(n, n) + cfg_.r_rate * D.transpose() * D;
  Eigen::VectorXd q = Su.transpose() * Q * offset - cfg_.r * dref - cfg_.r_rate * D.transpose() *
    d0;

  // constraints: [I; D] delta in [-max, max] x [d0 - rate dt, d0 + rate dt]
  Eigen::MatrixXd A(2 * n, n);
  A << Eigen::MatrixXd::Identity(n, n), D;
  Eigen::VectorXd l(2 * n), u(2 * n);
  const double dmax = cfg_.max_rate * cfg_.dt;
  for (int k = 0; k < n; ++k) {
    l(k) = -cfg_.max_delta;
    u(k) = cfg_.max_delta;
    l(n + k) = d0(k) - dmax;
    u(n + k) = d0(k) + dmax;
  }

  // ── OSQP ──
  Csc Pc(P, true), Ac(A, false);
  std::vector<c_float> qv(q.data(), q.data() + n), lv(l.data(), l.data() + 2 * n),
  uv(u.data(), u.data() + 2 * n);
  OSQPData data{};
  data.n = n;
  data.m = 2 * n;
  data.P = Pc.m;
  data.A = Ac.m;
  data.q = qv.data();
  data.l = lv.data();
  data.u = uv.data();
  OSQPSettings settings{};
  osqp_set_default_settings(&settings);
  settings.verbose = 0;
  settings.max_iter = cfg_.max_iter;
  settings.eps_abs = cfg_.eps;
  settings.eps_rel = cfg_.eps;
  settings.polish = 1;

  OSQPWorkspace * work = nullptr;
  const c_int err = osqp_setup(&work, &data, &settings);
  if (err != 0 || work == nullptr) {
    res.status = "setup failed";
  } else {
    osqp_solve(work);
    const c_int st = work->info->status_val;
    res.status = work->info->status;
    res.iterations = static_cast<int>(work->info->iter);
    res.ok = (st == OSQP_SOLVED || st == OSQP_SOLVED_INACCURATE);
    if (res.ok) {
      Eigen::Map<const Eigen::VectorXd> sol(work->solution->x, n);
      res.deltas.assign(sol.data(), sol.data() + n);
      res.delta = sol(0);
      const Eigen::VectorXd ep = Su * sol + offset;
      for (int k = 0; k < n; ++k) {
        res.e_y_pred.push_back(ep(2 * k));
      }
    }
    osqp_cleanup(work);
  }
  res.solve_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t0).count();
  return res;
}

double delta_to_carla_steer(double delta, double max_steer)
{
  const double s = -delta / (70.0 * M_PI / 180.0);
  return std::clamp(s, -max_steer, max_steer);
}

}  // namespace av_control_cpp
