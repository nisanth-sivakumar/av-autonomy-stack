"""
Minimal stand-in for the CARLA Python API, for offline testing only.

Implements just the calls used by av_planning and av_perception:
    Client(host, port).set_timeout() / get_world()
    World.get_map() / get_actors().filter(pattern)
    Map.name / get_waypoint(location, project_to_road, lane_type) / get_topology()
    Waypoint.transform / next(d) / get_left_lane() / get_right_lane() / lane_type
            / road_id / section_id / lane_id / is_junction / lane_width
    Actor.id / type_id / attributes / get_transform() / get_velocity() / bounding_box
    Location, Vector3D, LaneType.Driving

Everything is backed by offline_road (single-lane road, ROS frame); this
module converts to CARLA's left-handed frame (y and yaw negated) so the
nodes' own CARLA→ROS conversions are exercised.
"""

import fnmatch
import math

import offline_road as road


class LaneType:
    Driving = "Driving"


class Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


class BoundingBox:
    def __init__(self, extent):
        self.extent = extent


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Transform:
    def __init__(self, location, rotation=None):
        self.location = location
        self.rotation = rotation or Rotation()


class Waypoint:
    lane_type = LaneType.Driving
    # one road, one section, one lane (driving direction = increasing s)
    road_id, section_id, lane_id = 1, 0, -1
    is_junction = False
    lane_width = road.LANE_WIDTH

    def __init__(self, s):
        self.s = s
        x, y, yaw = road.pose_at(s)
        # ROS → CARLA: y and yaw negated
        self.transform = Transform(Location(x, -y, 0.0), Rotation(yaw=-math.degrees(yaw)))

    def next(self, distance):
        if self.s + distance > road.LENGTH:
            return []
        return [Waypoint(self.s + distance)]

    def get_left_lane(self):
        return None

    def get_right_lane(self):
        return None


class Map:
    name = "OfflineSimRoad"

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        s, _ = road.project(location.x, -location.y)
        return Waypoint(s)

    def get_topology(self):
        return [(Waypoint(0.0), Waypoint(road.LENGTH))]


class Actor:
    """A scenario car; its pose follows offline_road's shared time base."""

    def __init__(self, actor_id, type_id, spec, role_name=""):
        self.id = actor_id
        self.type_id = type_id
        self.attributes = {"role_name": role_name}
        self._spec = spec
        self.bounding_box = BoundingBox(Vector3D(spec["half_length"], spec["half_width"], 0.75))

    def _state(self):
        return road.obstacle_state(self._spec, road.elapsed())

    def get_transform(self):
        st = self._state()
        # ROS → CARLA: y and yaw negated
        return Transform(Location(st["x"], -st["y"], 0.0), Rotation(yaw=-math.degrees(st["yaw"])))

    def get_velocity(self):
        st = self._state()
        return Vector3D(st["vx"], -st["vy"], 0.0)


class ActorList(list):
    def filter(self, pattern):
        return [a for a in self if fnmatch.fnmatch(a.type_id, pattern)]


class World:
    def __init__(self):
        self._map = Map()
        self._actors = ActorList(
            Actor(100 + i, "vehicle.audi.a2", spec)
            for i, spec in enumerate(road.obstacle_specs())
        )

    def get_map(self):
        return self._map

    def get_actors(self):
        return self._actors


class Client:
    def __init__(self, host="localhost", port=2000):
        self._world = World()

    def set_timeout(self, seconds):
        pass

    def get_world(self):
        return self._world

    def get_server_version(self):
        return "offline"

    def get_client_version(self):
        return "offline"
