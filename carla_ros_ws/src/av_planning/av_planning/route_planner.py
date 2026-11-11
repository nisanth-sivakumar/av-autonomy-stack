"""
route_planner.py — global route search over the road topology (ROS-free).

The road network is a directed graph of lane segments: in CARLA, each
entry of `Map.get_topology()` is one (entry waypoint, exit waypoint) pair
for one lane of one road section, identified here by the key
(road_id, section_id, lane_id).  Segment B follows segment A when B's
entry is where A's exit is.

`RouteGraph.plan` runs A* from the segment the ego is on to the goal's
segment:

    g = sum of entered segment lengths (chord lengths, entry → exit)
    h = straight-line distance from the segment's exit to the goal
        segment's entry

h never overestimates (any chain of chords is at least the straight-line
distance), so the route is the shortest in chord length.  The planner
node then follows the route at every junction instead of taking the
first branch.
"""

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Hashable, Optional, Sequence


@dataclass(frozen=True)
class Segment:
    key: Hashable
    start: tuple[float, float]       # entry point (any consistent frame)
    end: tuple[float, float]         # exit point

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)


class RouteGraph:
    def __init__(self, segments: Sequence[Segment], connect_tol: float = 0.5):
        self.segments = {s.key: s for s in segments}
        self.successors: dict[Hashable, list[Hashable]] = {k: [] for k in self.segments}

        # Spatial hash of entry points so connecting is ~O(n)
        cell = max(connect_tol, 1e-3)
        grid: dict[tuple[int, int], list[Segment]] = {}
        for s in segments:
            grid.setdefault((int(s.start[0] // cell), int(s.start[1] // cell)), []).append(s)
        for a in segments:
            cx, cy = int(a.end[0] // cell), int(a.end[1] // cell)
            for dx, dy in itertools.product((-1, 0, 1), repeat=2):
                for b in grid.get((cx + dx, cy + dy), ()):
                    if b.key != a.key and math.dist(a.end, b.start) <= connect_tol:
                        self.successors[a.key].append(b.key)

    def plan(self, start_key: Hashable, goal_key: Hashable) -> Optional[list[Hashable]]:
        """Shortest segment sequence from start to goal (inclusive), or None."""
        if start_key not in self.segments or goal_key not in self.segments:
            return None
        if start_key == goal_key:
            return [start_key]
        goal_entry = self.segments[goal_key].start

        def h(key):
            return math.dist(self.segments[key].end, goal_entry)

        tie = itertools.count()
        open_heap = [(h(start_key), next(tie), start_key)]
        g = {start_key: 0.0}
        parent: dict[Hashable, Hashable] = {}
        closed = set()
        while open_heap:
            _, _, key = heapq.heappop(open_heap)
            if key == goal_key:
                route = [key]
                while key in parent:
                    key = parent[key]
                    route.append(key)
                return route[::-1]
            if key in closed:
                continue
            closed.add(key)
            for nxt in self.successors[key]:
                cand = g[key] + self.segments[nxt].length
                if cand < g.get(nxt, math.inf):
                    g[nxt] = cand
                    parent[nxt] = key
                    heapq.heappush(open_heap, (cand + h(nxt), next(tie), nxt))
        return None


def end_path_at_goal(
    path: Sequence[tuple[float, float, float]],
    goal: tuple[float, float],
    min_gap: float = 0.5,
) -> list[tuple[float, float, float]]:
    """
    End a path of (x, y, yaw) waypoints exactly at `goal`.

    The road-graph walk ends at the waypoint closest to the goal, which is up
    to half a step (1.5 m) before or past it, and the controller stops at the
    path's end — so the stop was bounded by the waypoint spacing.  Waypoints
    past the goal (along their own heading) or within `min_gap` of it are
    dropped, then the goal is appended, heading along the chord from the new
    last waypoint (or the dropped waypoint's heading if none is left).
    """
    out = list(path)
    yaw = out[-1][2] if out else 0.0
    gx, gy = goal
    while out:
        x, y, h = out[-1]
        past = (gx - x) * math.cos(h) + (gy - y) * math.sin(h) < 0.0
        if past or math.hypot(gx - x, gy - y) < min_gap:
            yaw = h
            out.pop()
        else:
            break
    if out:
        yaw = math.atan2(gy - out[-1][1], gx - out[-1][0])
    out.append((gx, gy, yaw))
    return out
