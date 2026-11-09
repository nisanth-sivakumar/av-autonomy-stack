// Parity tests for the C++ port of av_control/pure_pursuit.py's safety logic.
// Each mirrors a test in av_control/test/test_pure_pursuit.py.
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "av_control_cpp/safety.hpp"

using namespace av_control_cpp;  // NOLINT

namespace
{

std::vector<Point> line(double y = 0.0, int from = 1, int to = 40, double step = 1.0)
{
  std::vector<Point> p;
  for (int i = from; i < to; ++i) {
    p.push_back({i * step, y});
  }
  return p;
}

std::vector<Point> arc_path(double R, int n = 40)
{
  std::vector<Point> p;
  for (int i = 1; i < n; ++i) {
    const double a = i / R;
    p.push_back({R * std::sin(a), R * (1 - std::cos(a))});
  }
  return p;
}

Obstacle car(double x, double y, double vx = 0.0, double vy = 0.0, double yaw = 0.0)
{
  return {x, y, yaw, 2.2, 0.9, vx, vy};
}

CorridorParams corridor(double heading = 0.0)
{
  CorridorParams p;
  p.corridor_half_width = 1.35;
  p.ego_half_length = 2.4;
  p.heading_check_m = heading;
  return p;
}

}  // namespace

TEST(Prune, DropsWaypointsBehindAndReturnsTheLast)
{
  std::vector<Point> w{{-2, 0}, {-1, 0}, {1, 0}, {2, 0}};
  auto last = prune_passed(w, 0, 0, 0);
  ASSERT_TRUE(last.has_value());
  EXPECT_DOUBLE_EQ(last->x, -1);
  ASSERT_EQ(w.size(), 2u);
  EXPECT_DOUBLE_EQ(w[0].x, 1);
}

TEST(Corridor, BoxGapOnStraight)
{
  auto hit = obstacle_along_path(0, 0, 0, line(), {car(10, 0)}, corridor());
  ASSERT_TRUE(hit);
  EXPECT_NEAR(hit->first, 10 - 2.2 - 2.4, 1e-9);
}

TEST(Corridor, BoxWidthCounts)
{
  // centre 2.0 m off, half width 0.9 → edge at 1.1 < 1.35
  EXPECT_TRUE(obstacle_along_path(0, 0, 0, line(), {car(10, 2.0)}, corridor()));
  EXPECT_FALSE(obstacle_along_path(0, 0, 0, line(), {car(10, 2.4)}, corridor()));
}

TEST(Corridor, CrossingPedestrianPredictedIn)
{
  Obstacle ped{10, 4.0, 0.0, 0.3, 0.3, 0.0, -1.4};
  EXPECT_TRUE(obstacle_along_path(0, 0, 0, line(), {ped}, corridor()));
}

TEST(Corridor, SlowLeadReportsRawSpeed)
{
  auto hit = obstacle_along_path(0, 0, 0, line(0, 1, 30), {car(10, 0, 0.4)}, corridor());
  ASSERT_TRUE(hit);
  EXPECT_NEAR(hit->second, 0.4, 1e-9);
}

TEST(Corridor, HeadingCorridorCatchesLeadThePlanLeft)
{
  std::vector<Point> plan;
  for (int x = 2; x < 60; x += 3) {
    plan.push_back({static_cast<double>(x), 3.5});
  }
  const auto lead = car(6.5, 0.0, 1.0);
  EXPECT_FALSE(obstacle_along_path(0, 1.5, 0, plan, {lead}, corridor()));
  auto hit = obstacle_along_path(0, 1.5, 0, plan, {lead}, corridor(4.5));
  ASSERT_TRUE(hit);
  EXPECT_NEAR(hit->first, 6.5 - 2.2 - 2.4, 1e-9);
  EXPECT_FALSE(obstacle_along_path(0, 3.3, 0, plan, {lead}, corridor(4.5)));
}

TEST(Curvature, SustainedOnArcZeroForLaneChange)
{
  std::vector<Point> a{{0, 0}};
  for (auto p : arc_path(26.5)) {
    a.push_back(p);
  }
  EXPECT_NEAR(sustained_curvature(a), 1 / 26.5, 0.1 / 26.5);
  std::vector<Point> s{{0, 0}};
  for (int x = 1; x <= 20; ++x) {
    const double f = x / 20.0;
    s.push_back({static_cast<double>(x), 3.5 * (3 * f * f - 2 * f * f * f)});
  }
  for (int x = 21; x < 40; ++x) {
    s.push_back({static_cast<double>(x), 3.5});
  }
  EXPECT_DOUBLE_EQ(sustained_curvature(s), 0.0);
}

TEST(Corridor, OncomingCarOnCurvePredictedAlongItsLane)
{
  const double R = 30.0;
  const double a = 25.0 / R;
  const double ox = (R - 3.5) * std::sin(a), oy = R - (R - 3.5) * std::cos(a);
  const double yaw = a + M_PI;
  Obstacle onc{ox, oy, yaw, 2.2, 0.9, 4 * std::cos(yaw), 4 * std::sin(yaw)};
  auto p = corridor(4.5);
  p.ego_speed = 2.0;
  EXPECT_FALSE(obstacle_along_path(0, 0, 0, arc_path(R, 60), {onc}, p));
  Obstacle cut_in{10.0, 3.0, -0.5, 2.2, 0.9, 1.0, -1.0};
  EXPECT_TRUE(obstacle_along_path(0, 0, 0, line(), {cut_in}, p));
}

TEST(Speed, LimitAndCommand)
{
  EXPECT_DOUBLE_EQ(obstacle_speed_limit(std::nullopt, 3.0, 3.5, 2.0), 3.0);
  EXPECT_DOUBLE_EQ(obstacle_speed_limit(3.0, 3.0, 3.5, 2.0), 0.0);
  EXPECT_NEAR(obstacle_speed_limit(4.5, 3.0, 3.5, 2.0), 2.0, 1e-12);
  auto [thr, brk] = speed_command(0.0, 0.0, 0.5, 0.5);
  EXPECT_DOUBLE_EQ(thr, 0.0);
  EXPECT_DOUBLE_EQ(brk, 1.0);
  auto [t2, b2] = speed_command(2.8, 3.0, 0.5, 0.5);
  EXPECT_NEAR(t2, 0.1, 1e-12);
  EXPECT_DOUBLE_EQ(b2, 0.0);
}

TEST(Governor, StopAndHoldAgainstFlicker)
{
  ObstacleSpeedGovernor g(3.5, 2.0, 2.0, 5);
  for (int i = 0; i < 6; ++i) {
    g.update(3.4, 3.0);
  }
  EXPECT_TRUE(g.holding());
  // one clear frame must not release
  EXPECT_DOUBLE_EQ(g.update(std::nullopt, 3.0), 0.0);
}

TEST(Governor, FollowsMovingLeadAtTimeGap)
{
  ObstacleSpeedGovernor g(3.5, 2.0);
  double v = 0.0;
  for (int i = 0; i < 10; ++i) {
    v = g.update(5.0, 3.0, 1.5);
  }
  // d_des = 3.5 + 1.5 = 5.0 → v = v_lead
  EXPECT_NEAR(v, 1.5, 1e-9);
  EXPECT_FALSE(g.holding());
}

TEST(Governor, KeepsFollowingSlowNoisyLead)
{
  // Same rate test as the Python one (different RNG): a 0.6 m/s noisy lead
  // stops the car in <= 10 % of seeds (92 % without hysteresis).
  int stopped = 0;
  for (int seed = 0; seed < 100; ++seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> gap_noise(0.0, 0.4), v_noise(0.0, 0.2);
    ObstacleSpeedGovernor g(3.5, 1.5);
    double gap = 5.0, v = 0.6;
    for (int i = 0; i < 600; ++i) {
      const double cmd = g.update(gap + gap_noise(rng), 3.0, 0.6 + v_noise(rng));
      v += std::clamp(cmd - v, -0.15, 0.15);
      gap += (0.6 - v) * 0.1;
      if (v <= 0.05) {
        ++stopped;
        break;
      }
    }
  }
  EXPECT_LE(stopped, 10);
}

TEST(Governor, ExactParityWithPython)
{
  // test_pure_pursuit.py::test_governor_parity_reference, same inputs
  ObstacleSpeedGovernor g(3.5, 1.5);
  double gap = 5.0, v = 0.6, c = 0.0;
  for (int i = 0; i < 600; ++i) {
    c = g.update(gap + 0.4 * std::sin(1.7 * i), 3.0, 0.6 + 0.2 * std::sin(2.3 * i + 1.0));
    v += std::clamp(c - v, -0.15, 0.15);
    gap += (0.6 - v) * 0.1;
  }
  EXPECT_NEAR(c, 0.6192893695862955, 1e-12);
  EXPECT_NEAR(gap, 4.108379436166265, 1e-12);
}
