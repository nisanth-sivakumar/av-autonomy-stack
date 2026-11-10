#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "av_control_cpp/mpc.hpp"
#include "av_control_cpp/safety.hpp"

using av_control_cpp::back_from_previous;
using av_control_cpp::build_reference;
using av_control_cpp::delta_to_carla_steer;
using av_control_cpp::extend_back;
using av_control_cpp::LateralMpc;
using av_control_cpp::linearize;
using av_control_cpp::MpcConfig;
using av_control_cpp::Point;

namespace
{

std::vector<Point> straight(double y = 0.0, double len = 80.0, double step = 3.0)
{
  std::vector<Point> p;
  for (double x = -6.0; x <= len; x += step) {
    p.push_back({x, y});
  }
  return p;
}

// left arc from (0, 0) heading +x
std::vector<Point> arc(double radius, double len = 120.0, double step = 3.0)
{
  std::vector<Point> p{{-6.0, 0.0}, {-3.0, 0.0}};
  for (double s = 0.0; s <= len; s += step) {
    const double a = s / radius;
    p.push_back({radius * std::sin(a), radius * (1.0 - std::cos(a))});
  }
  return p;
}

// Rear-axle kinematic bicycle, the model the MPC assumes (and the offline sim).
struct Car
{
  double x, y, yaw, v;
  void step(double delta, double dt, double wheelbase)
  {
    yaw += v / wheelbase * std::tan(delta) * dt;
    x += v * std::cos(yaw) * dt;
    y += v * std::sin(yaw) * dt;
  }
};

}  // namespace

// ── Model and linearization ──

TEST(Model, LinearizationOnStraight)
{
  MpcConfig cfg;
  const auto m = linearize(3.0, 0.0, cfg);
  EXPECT_DOUBLE_EQ(m.delta_ref, 0.0);
  EXPECT_DOUBLE_EQ(m.a12, cfg.dt * 3.0);
  EXPECT_DOUBLE_EQ(m.b2, cfg.dt * 3.0 / cfg.wheelbase);
}

TEST(Model, FeedForwardFollowsCurvature)
{
  MpcConfig cfg;
  const double kappa = 1.0 / 14.0;   // CARLA junction turn
  const auto m = linearize(3.0, kappa, cfg);
  EXPECT_NEAR(m.delta_ref, std::atan(cfg.wheelbase * kappa), 1e-12);
  // sec^2 scaling: the yaw-rate sensitivity grows with the wheel angle
  EXPECT_GT(m.b2, cfg.dt * 3.0 / cfg.wheelbase);
  // the model's yaw rate at delta_ref equals v * kappa (the path's)
  EXPECT_NEAR(3.0 * std::tan(m.delta_ref) / cfg.wheelbase, 3.0 * kappa, 1e-12);
}

TEST(Model, SlowSpeedKeepsAuthority)
{
  MpcConfig cfg;
  EXPECT_DOUBLE_EQ(linearize(0.0, 0.0, cfg).b2, linearize(cfg.min_model_speed, 0.0, cfg).b2);
}

// ── Reference ──

TEST(Reference, ErrorsOnStraight)
{
  MpcConfig cfg;
  const auto r = build_reference(straight(), 0.0, 0.5, 0.1, 3.0, cfg);
  ASSERT_TRUE(r.valid);
  EXPECT_NEAR(r.e_y, 0.5, 1e-9);     // + = left of the path
  EXPECT_NEAR(r.e_psi, 0.1, 1e-9);
  for (double k : r.kappa) {
    EXPECT_NEAR(k, 0.0, 1e-9);
  }
}

TEST(Reference, CurvatureOnArc)
{
  MpcConfig cfg;
  const auto r = build_reference(arc(30.0), 0.0, 0.0, 0.0, 3.0, cfg);
  ASSERT_TRUE(r.valid);
  // samples past the first vertex are on the arc
  EXPECT_NEAR(r.kappa.back(), 1.0 / 30.0, 0.002);
}

TEST(Reference, InvalidWithoutPath)
{
  MpcConfig cfg;
  EXPECT_FALSE(build_reference({{0.0, 0.0}}, 0, 0, 0, 3.0, cfg).valid);
}

TEST(Reference, CurvatureIsZeroBeyondThePathEnd)
{
  MpcConfig cfg;
  const auto r = build_reference(arc(30.0, 6.0), 0.0, 0.0, 0.0, 3.0, cfg);
  ASSERT_TRUE(r.valid);
  EXPECT_DOUBLE_EQ(r.kappa.back(), 0.0);
}

// ── Solver and closed loop ──

TEST(Mpc, ZeroErrorOnStraightGivesZeroSteer)
{
  LateralMpc mpc;
  const auto r = build_reference(straight(), 0.0, 0.0, 0.0, 3.0, mpc.config());
  const auto res = mpc.solve(r, 3.0, 0.0);
  ASSERT_TRUE(res.ok) << res.status;
  EXPECT_NEAR(res.delta, 0.0, 1e-4);
}

TEST(Mpc, SteersTowardThePath)
{
  LateralMpc mpc;
  // car 1 m left of the path → steer right (negative wheel angle)
  const auto r = build_reference(straight(), 0.0, 1.0, 0.0, 3.0, mpc.config());
  const auto res = mpc.solve(r, 3.0, 0.0);
  ASSERT_TRUE(res.ok);
  EXPECT_LT(res.delta, 0.0);
  // the predicted lateral error shrinks over the horizon
  EXPECT_LT(std::abs(res.e_y_pred.back()), 0.5);
}

TEST(Mpc, ClosedLoopConvergesOntoStraight)
{
  LateralMpc mpc;
  const auto& cfg = mpc.config();
  Car car{0.0, 1.5, 0.15, 3.0};       // 1.5 m off, heading away
  double delta = 0.0, max_abs_after = 0.0;
  const double ctrl_dt = 0.05;
  for (int i = 0; i < 400; ++i) {     // 20 s
    const auto r = build_reference(straight(0.0, 200.0), car.x, car.y, car.yaw, car.v, cfg);
    const auto res = mpc.solve(r, car.v, delta);
    ASSERT_TRUE(res.ok);
    delta = res.delta;
    car.step(delta, ctrl_dt, cfg.wheelbase);
    if (i > 200) {
      max_abs_after = std::max(max_abs_after, std::abs(car.y));
    }
  }
  EXPECT_LT(max_abs_after, 0.05);
}

TEST(Mpc, ClosedLoopTracksJunctionCurve)
{
  // R 14 m (CARLA junction turns): Pure Pursuit cut these by up to 0.4 m
  LateralMpc mpc;
  const auto& cfg = mpc.config();
  const double R = 14.0;
  Car car{-3.0, 0.0, 0.0, 3.0};
  double delta = 0.0, max_err = 0.0;
  for (int i = 0; i < 300; ++i) {     // 15 s = 45 m, half the arc
    const auto path = arc(R, 60.0, 1.0);
    const auto r = build_reference(path, car.x, car.y, car.yaw, car.v, cfg);
    const auto res = mpc.solve(r, car.v, delta);
    ASSERT_TRUE(res.ok);
    delta = res.delta;
    car.step(delta, 0.05, cfg.wheelbase);
    if (car.x > 2.0) {
      const double err = std::abs(std::hypot(car.x, car.y - R) - R);
      max_err = std::max(max_err, err);
    }
  }
  EXPECT_LT(max_err, 0.15);
}

TEST(Mpc, RespectsSteeringAndRateLimits)
{
  LateralMpc mpc;
  const auto& cfg = mpc.config();
  // a large error: wants a big wheel angle, from straight wheels
  const auto r = build_reference(straight(), 0.0, 3.0, 0.0, 3.0, cfg);
  const auto res = mpc.solve(r, 3.0, 0.0);
  ASSERT_TRUE(res.ok);
  double prev = 0.0;
  for (double d : res.deltas) {
    EXPECT_LE(std::abs(d), cfg.max_delta + 1e-6);
    EXPECT_LE(std::abs(d - prev), cfg.max_rate * cfg.dt + 1e-6);
    prev = d;
  }
  // the rate limit binds on the first step
  EXPECT_NEAR(std::abs(res.delta), cfg.max_rate * cfg.dt, 1e-4);
}

TEST(Mpc, SolverFailureIsReported)
{
  // previous wheel angle far outside the limits: |delta| <= max and
  // |delta - prev| <= rate*dt cannot both hold → primal infeasible
  LateralMpc mpc;
  const auto r = build_reference(straight(), 0.0, 0.0, 0.0, 3.0, mpc.config());
  const auto res = mpc.solve(r, 3.0, 3.0);
  EXPECT_FALSE(res.ok);
  EXPECT_FALSE(res.status.empty());
  EXPECT_TRUE(res.deltas.empty());
}

TEST(Mpc, InvalidReferenceIsAFailure)
{
  LateralMpc mpc;
  av_control_cpp::Reference bad;
  const auto res = mpc.solve(bad, 3.0, 0.0);
  EXPECT_FALSE(res.ok);
  EXPECT_EQ(res.status, "invalid reference");
}

TEST(Mpc, SolveIsFast)
{
  LateralMpc mpc;
  const auto r = build_reference(arc(30.0), 0.0, 0.3, 0.05, 3.0, mpc.config());
  double worst = 0.0;
  for (int i = 0; i < 50; ++i) {
    worst = std::max(worst, mpc.solve(r, 3.0, 0.0).solve_ms);
  }
  EXPECT_LT(worst, 10.0);   // control period is 50 ms
}

// ── CARLA steer mapping ──

TEST(Steer, SignAndScale)
{
  const double rad70 = 70.0 * M_PI / 180.0;
  EXPECT_NEAR(delta_to_carla_steer(0.35, 0.6), -0.35 / rad70, 1e-12);   // left wheel → negative
  EXPECT_DOUBLE_EQ(delta_to_carla_steer(-5.0, 0.6), 0.6);                // clamped
}

TEST(Reference, HeadingErrorIsContinuousAlongChords)
{
  // a car exactly on an R 30 m arc, sampled with 3 m chords: e_psi must stay
  // ~0 everywhere (a chord's own heading is off by up to half the 5.7 deg
  // vertex turn, which stepped the steering)
  MpcConfig cfg;
  const double R = 30.0;
  const auto path = arc(R, 60.0, 3.0);
  double worst = 0.0;
  for (double s = 3.0; s < 45.0; s += 0.25) {
    const double a = s / R;
    const auto r = build_reference(path, R * std::sin(a), R * (1 - std::cos(a)), a, 3.0, cfg);
    ASSERT_TRUE(r.valid);
    worst = std::max(worst, std::abs(r.e_psi));
  }
  EXPECT_LT(worst, 0.01);   // rad; chord heading alone gives up to 0.05
}

TEST(Reference, ExtendBackFollowsTheArc)
{
  const double R = 30.0;
  std::vector<Point> path;
  for (double s = 9.0; s < 40.0; s += 3.0) {
    path.push_back({R * std::sin(s / R), R * (1 - std::cos(s / R))});
  }
  const auto ext = extend_back(path, 1.0 / R, 6.0, 1.5);
  ASSERT_EQ(ext.size(), path.size() + 4);
  for (std::size_t i = 0; i < 4; ++i) {   // back points lie on the circle
    EXPECT_NEAR(std::hypot(ext[i].x, ext[i].y - R), R, 0.01);
  }
  // and reach 6 m back along it: s = 3 m
  EXPECT_NEAR(ext[0].x, R * std::sin(3.0 / R), 0.05);
}

TEST(Reference, ExtendBackOnStraightIsStraight)
{
  const auto ext = extend_back({{5, 1}, {8, 1}}, 0.0, 3.0, 1.5);
  ASSERT_EQ(ext.size(), 4u);
  EXPECT_NEAR(ext[0].x, 2.0, 1e-9);
  EXPECT_NEAR(ext[0].y, 1.0, 1e-9);
}

TEST(Reference, CurvatureAtTheFirstSegmentIsTheArcs)
{
  // rear axle on the first segment of a path that starts on an arc: the
  // feed-forward must use the arc's curvature, not 0
  MpcConfig cfg;
  const double R = 30.0;
  std::vector<Point> path;
  for (double s = 9.0; s < 60.0; s += 3.0) {
    path.push_back({R * std::sin(s / R), R * (1 - std::cos(s / R))});
  }
  const double a = 9.5 / R;
  const auto r = build_reference(path, R * std::sin(a), R * (1 - std::cos(a)), a, 3.0, cfg);
  ASSERT_TRUE(r.valid);
  EXPECT_NEAR(r.kappa[0], 1.0 / R, 0.002);
}


// ── Reference behind the plan's start (from the previous plans) ──

namespace
{

// CARLA-like junction: 30 m straight, 90 deg left turn R 14 m, straight
void road_pose(double s, double & x, double & y, double & h)
{
  const double s1 = 30.0, r = 14.0, arc_len = r * M_PI / 2.0;
  if (s <= s1) {
    x = s; y = 0.0; h = 0.0;
  } else if (s <= s1 + arc_len) {
    const double a = (s - s1) / r;
    x = s1 + r * std::sin(a); y = r * (1.0 - std::cos(a)); h = a;
  } else {
    x = s1 + r; y = r + (s - s1 - arc_len); h = M_PI / 2.0;
  }
}

// The planner's path: road-graph vertices every 3 m, starting at the first
// one ahead of the vehicle centre (as in CARLA: 0-3 m ahead)
std::vector<Point> grid_plan(double s_centre, double len = 60.0)
{
  std::vector<Point> p;
  for (double sv = std::floor(s_centre / 3.0) * 3.0 + 3.0; sv <= s_centre + len; sv += 3.0) {
    double x, y, h;
    road_pose(sv, x, y, h);
    p.push_back({x, y});
  }
  return p;
}

struct Worst
{
  double e_y{0.0};
  double e_psi{0.0};
};

// Drive the rear axle exactly along the road and record the reference error
// the MPC would see; `history` selects the trail from previous plans.
Worst drive_through_turn(bool history)
{
  MpcConfig cfg;
  std::vector<Point> back, plan;
  Worst w;
  for (double sr = 2.0; sr < 75.0; sr += 0.15) {
    double rx, ry, h;
    road_pose(sr, rx, ry, h);
    const std::vector<Point> next = grid_plan(sr + 1.5);
    std::vector<Point> path;
    if (history) {
      std::vector<Point> prev(back);
      prev.insert(prev.end(), plan.begin(), plan.end());
      back = back_from_previous(prev, next);
      plan = next;
      path = back;
      path.insert(path.end(), plan.begin(), plan.end());
      if (back.empty()) {
        path = extend_back(plan, 0.0, 6.0);     // first plan: on the straight
      }
    } else {
      path = extend_back(next, av_control_cpp::sustained_curvature(next), 6.0);
    }
    const auto ref = build_reference(path, rx, ry, h, 3.0, cfg);
    EXPECT_TRUE(ref.valid);
    w.e_y = std::max(w.e_y, std::abs(ref.e_y));
    w.e_psi = std::max(w.e_psi, std::abs(ref.e_psi));
  }
  return w;
}

}  // namespace

TEST(Reference, BackFromPreviousOnTheSameLineIsExact)
{
  const auto line = arc(14.0, 60.0);
  const std::vector<Point> next(line.begin() + 6, line.end());
  const auto back = back_from_previous(line, next, 10.0);
  ASSERT_GE(back.size(), 3u);
  for (std::size_t i = 0; i < back.size(); ++i) {
    const Point & want = line[6 - back.size() + i];
    EXPECT_NEAR(back[i].x, want.x, 1e-9);
    EXPECT_NEAR(back[i].y, want.y, 1e-9);
  }
}

TEST(Reference, BackFromPreviousFollowsASidewaysReplan)
{
  // a lattice replan 0.3 m to the left of the previous one, same shape
  const auto prev = straight(0.0);
  auto next = straight(0.3);
  next.erase(next.begin(), next.begin() + 4);
  const auto back = back_from_previous(prev, next, 6.0);
  ASSERT_FALSE(back.empty());
  for (const auto & p : back) {
    EXPECT_NEAR(p.y, 0.3, 1e-9);   // moved with the new plan: no kink at the join
    EXPECT_LT(p.x, next[0].x);
  }
}

TEST(Reference, BackFromPreviousRejectsANewRoute)
{
  const auto prev = straight(0.0);
  auto far = straight(3.5);                       // a lane away
  EXPECT_TRUE(back_from_previous(prev, far).empty());
  std::vector<Point> crossing{{10.0, 0.0}, {10.0, 3.0}, {10.0, 6.0}};   // 90 deg off
  EXPECT_TRUE(back_from_previous(prev, crossing).empty());
}

TEST(Reference, ContinuousThroughCurveEntryAndExit)
{
  // Regression for CARLA Step 6: extrapolating the plan backwards read
  // 20-30 deg heading errors on a car that was exactly on the road.
  const Worst old = drive_through_turn(false);
  const Worst now = drive_through_turn(true);
  std::printf("old: e_y %.3f m, e_psi %.3f rad; trail: e_y %.3f m, e_psi %.3f rad\n",
              old.e_y, old.e_psi, now.e_y, now.e_psi);
  EXPECT_GT(old.e_psi, 0.2);                       // the old extension's failure
  EXPECT_LT(now.e_y, 0.05);
  // what remains is the 3 m polyline itself: where the straight meets the
  // arc, the vertex heading (mean of the two chords) is a quarter of a chord's
  // turn (3/14 rad) off the road's
  EXPECT_LT(now.e_psi, 0.08);
}
