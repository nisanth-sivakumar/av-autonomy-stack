"""
obstacles.py — obstacle model and clearance geometry shared by the planners.

An obstacle is an oriented box with a velocity (ROS map frame):

    (x, y, yaw, half_length, half_width, vx, vy)

Clearance
---------
`box_clearance` measures, for each path point (the ego's centre when it is
there) and each obstacle, the gap between the ego and the obstacle box in
the path's frame:

    along-path gap  = max(0, |along| − obstacle half-extent along the path − ego half-length
                             − long_margin)
    lateral gap     = max(0, |lat|   − obstacle half-extent across the path)
    clearance       = hypot(along-path gap, lateral gap)

When the ego's footprint overlaps the obstacle lengthwise (along-path gap
0), the clearance is the lateral distance from the ego's centreline to the
obstacle's edge, so the footprints touch when it drops below the ego's
half-width (1.08 m for CARLA's Model 3, mirrors included).  Planner margins
are expressed in the same terms.

`long_margin` extends the stretch that must be laterally clear to a few
metres before and after the obstacle.  Without it only the points where
the footprints overlap are penalised, so a smooth planner (MPPI) starts
its swerve too late to reach full offset — in CARLA it passed a parked
car with 0.16 m to spare.

Moving obstacles
----------------
With `times` (seconds until the ego reaches each path point), each moving
obstacle is first moved to where it will be then: o + v·t.
  - Obstacles slower than `moving_threshold` are treated as stationary, so
    velocity noise on parked cars is not extrapolated into drift.
  - For moving obstacles, path points the ego reaches after `max_time` are
    ignored: the prediction is not trustworthy that far out, and freezing
    the obstacle at its `max_time` position would put a phantom obstacle
    on the path (e.g. a lead car pulling away).  The planners replan every
    tick, so those points are checked again as the ego gets closer.

Followed vs avoided
-------------------
`avoidable` drops obstacles the planners should not steer around:
  - moving the same way as the path (a lead car): without lane-change
    logic the planners don't steer around traffic, the controller follows;
  - impassable: even at the planner's maximum lateral offset the clearance
    would stay below the controller's stop corridor, so the controller
    stops for it anyway and swerving would only leave the car angled
    (seen in CARLA with a car parked across the lane).
Parked, oncoming and crossing obstacles that can be passed are kept.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

# Default size assumed for obstacles reported without extent (a mid-size car)
DEFAULT_HALF_LENGTH = 2.2
DEFAULT_HALF_WIDTH = 0.9


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    yaw: float = 0.0
    half_length: float = DEFAULT_HALF_LENGTH
    half_width: float = DEFAULT_HALF_WIDTH
    vx: float = 0.0
    vy: float = 0.0

    def as_tuple(self) -> tuple:
        return (self.x, self.y, self.yaw, self.half_length, self.half_width, self.vx, self.vy)


def as_obstacle_array(obstacles: Optional[Iterable]) -> np.ndarray:
    """
    Normalise obstacles to an (N, 7) array.

    Accepts `Obstacle`s, 7-tuples, or bare (x, y) points.  Bare points get
    zero extent (a point obstacle), which keeps the geometry exact for
    callers that really mean a point.
    """
    rows = []
    for o in obstacles or ():
        if isinstance(o, Obstacle):
            rows.append(o.as_tuple())
        elif len(o) == 2:
            rows.append((o[0], o[1], 0.0, 0.0, 0.0, 0.0, 0.0))
        elif len(o) == 7:
            rows.append(tuple(o))
        else:
            raise ValueError(f"obstacle must be Obstacle, (x, y) or 7-tuple, got {o!r}")
    return np.asarray(rows, dtype=float).reshape(-1, 7)


def path_times(xs: Sequence[float], ys: Sequence[float], ego_xy: Optional[tuple],
               ego_speed: float, min_speed: float = 1.0) -> np.ndarray:
    """
    Time for the ego to reach each path point: arc length from the ego
    divided by max(ego_speed, min_speed).
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if len(xs) == 0:
        return np.zeros(0)
    x0, y0 = ego_xy if ego_xy is not None else (xs[0], ys[0])
    seg = np.hypot(np.diff(np.concatenate([[x0], xs])), np.diff(np.concatenate([[y0], ys])))
    s = np.cumsum(seg)
    return s / max(ego_speed, min_speed)


def box_clearance(xs, ys, yaws, obs: np.ndarray, ego_half_length: float,
                  times: Optional[np.ndarray] = None, max_time: float = 6.0,
                  moving_threshold: float = 0.5, long_margin: float = 0.0) -> np.ndarray:
    """
    Clearance between the ego (centred on each path point) and each
    obstacle box, shape (..., H, N).  See module docstring.

    xs, ys, yaws : (..., H) path points and headings
    obs          : (N, 7) from as_obstacle_array
    times        : (H,) or (..., H) seconds until the ego reaches each point
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    yaws = np.broadcast_to(np.asarray(yaws, float), xs.shape)
    ox, oy, oyaw, hl, hw, vx, vy = (obs[:, i] for i in range(7))
    moving = np.hypot(vx, vy) >= moving_threshold                    # (N,)

    beyond = None
    if times is not None:
        t = np.broadcast_to(np.asarray(times, float), xs.shape)[..., None]
        ox = ox + np.where(moving, vx, 0.0) * t
        oy = oy + np.where(moving, vy, 0.0) * t
        beyond = (t > max_time) & moving                              # (..., H, N)

    c, s = np.cos(yaws)[..., None], np.sin(yaws)[..., None]
    dx = xs[..., None] - ox
    dy = ys[..., None] - oy
    along = dx * c + dy * s
    lat = -dx * s + dy * c

    rel = oyaw - yaws[..., None]
    cr, sr = np.abs(np.cos(rel)), np.abs(np.sin(rel))
    ext_along = hl * cr + hw * sr
    ext_lat = hl * sr + hw * cr

    gap_along = np.maximum(0.0, np.abs(along) - ext_along - ego_half_length - long_margin)
    gap_lat = np.maximum(0.0, np.abs(lat) - ext_lat)
    clearance = np.hypot(gap_along, gap_lat)
    if beyond is not None:
        clearance = np.where(beyond, np.inf, clearance)
    return clearance


def avoidable(obs: np.ndarray, path_xy: np.ndarray, path_yaw: np.ndarray,
              moving_threshold: float = 0.5, max_offset: Optional[float] = None,
              pass_clearance: Optional[float] = None) -> np.ndarray:
    """
    Obstacles the planners should steer around (see module docstring).

    max_offset     : the planner's largest lateral offset from the path
    pass_clearance : clearance below which the controller stops (its corridor);
                     with max_offset, obstacles that cannot be passed with at
                     least this clearance are dropped
    """
    if len(obs) == 0 or len(path_xy) == 0:
        return obs
    d = np.hypot(path_xy[None, :, 0] - obs[:, None, 0], path_xy[None, :, 1] - obs[:, None, 1])
    nearest = d.argmin(axis=1)
    yaw = np.asarray(path_yaw)[nearest]
    c, s = np.cos(yaw), np.sin(yaw)
    keep = obs[:, 5] * c + obs[:, 6] * s < moving_threshold
    if max_offset is not None and pass_clearance is not None:
        px, py = path_xy[nearest, 0], path_xy[nearest, 1]
        lat = -(obs[:, 0] - px) * s + (obs[:, 1] - py) * c
        rel = obs[:, 2] - yaw
        ext_lat = obs[:, 3] * np.abs(np.sin(rel)) + obs[:, 4] * np.abs(np.cos(rel))
        best_clearance = np.abs(lat) + max_offset - ext_lat
        keep &= best_clearance >= pass_clearance
    return obs[keep]
