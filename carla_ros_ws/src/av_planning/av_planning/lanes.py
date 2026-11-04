"""
lanes.py — neighbouring lanes along the route (ROS-free, duck-typed waypoints).

Works on CARLA-style waypoint objects (`get_left_lane()`, `get_right_lane()`,
`lane_id`, `lane_type`, `transform.location`, `transform.rotation.yaw`)
so it can be unit-tested with stubs.

Direction
---------
In OpenDRIVE (and CARLA) the sign of `lane_id` says which side of the
road's reference line a lane is on, and so its driving direction: lanes
with the same sign as the ego's lane run the same way, lanes with the
opposite sign are oncoming.  CARLA's `get_left_lane()` is relative to the
lane's own driving direction, so the left neighbour of a right-hand-
traffic lane next to the centre line is the first oncoming lane.

A neighbour lane is reported only if it exists (as a driving lane) for
the first `min_length` waypoints of the route; it is truncated where it
stops existing (lane ends, junction).
"""

from typing import Callable, Optional, Sequence

import numpy as np

from av_planning.behavior import ONCOMING, SAME, Lane


def lane_direction(ego_lane_id: int, other_lane_id: int) -> int:
    """SAME if both lanes run the same way (same lane_id sign), else ONCOMING."""
    return SAME if (ego_lane_id > 0) == (other_lane_id > 0) else ONCOMING


def wp_pose(wp) -> tuple[float, float, float]:
    """ROS-frame (x, y, yaw) of a CARLA-style waypoint."""
    loc, rot = wp.transform.location, wp.transform.rotation
    return loc.x, -loc.y, -np.radians(rot.yaw)


def neighbour_waypoints(route: Sequence, side: str,
                        is_driving: Callable[[object], bool]) -> list:
    """
    The `side` ('left' / 'right') neighbour of each route waypoint, up to the
    first waypoint that has none (or a non-driving one).
    """
    out = []
    for wp in route:
        n = wp.get_left_lane() if side == "left" else wp.get_right_lane()
        if n is None or not is_driving(n):
            break
        out.append(n)
    return out


def neighbour_lane(route: Sequence, side: str, is_driving: Callable[[object], bool],
                   min_length: int = 5, width: Optional[float] = None) -> tuple[Optional[Lane], list]:
    """
    (Lane, its waypoints) for the `side` neighbour along the route, or
    (None, []) if it doesn't exist for at least `min_length` waypoints.

    The Lane's centre line runs in the *ego's* travel direction (an oncoming
    lane's waypoints point the other way; their positions are used as-is,
    with the ego route's heading), so arc lengths and lateral offsets are
    comparable with the ego lane.
    """
    if not route:
        return None, []
    wps = neighbour_waypoints(route, side, is_driving)
    if len(wps) < min_length:
        return None, []
    direction = lane_direction(route[0].lane_id, wps[0].lane_id)
    centre = []
    for ego_wp, n in zip(route, wps):
        x, y, _ = wp_pose(n)
        _, _, ego_yaw = wp_pose(ego_wp)
        centre.append((x, y, ego_yaw))
    w = width if width is not None else float(getattr(wps[0], "lane_width", 3.5))
    return Lane(np.asarray(centre, float), direction, w), wps
