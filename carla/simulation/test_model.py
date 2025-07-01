import carla
import subprocess
import time
import psutil
import random

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32
from carla_msgs.msg import CarlaEgoVehicleStatus

from navigation.local_planner import RoadOption
from navigation.global_route_planner import GlobalRoutePlanner
from navigation.global_route_planner_dao import GlobalRoutePlannerDAO

# === Constants ===
VEHICLE_TYPE = 'vehicle.nissan.patrol'
MAP = 'Mine_01'
SAMPLING_RESOLUTION = 1.0

COMMAND_MAP = {
    RoadOption.VOID: -1,
    RoadOption.LEFT: 1,
    RoadOption.RIGHT: 2,
    RoadOption.STRAIGHT: 3,
    RoadOption.LANEFOLLOW: 4,
    RoadOption.CHANGELANELEFT: 5,
    RoadOption.CHANGELANERIGHT: 6
}


# === Utility Functions ===
def launch_carla():
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] and 'CarlaUnreal' in proc.info['name']:
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return subprocess.Popen(
        ["./CarlaUnreal.sh", "--ros2", "-RenderOffScreen"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def get_closest_command(ego_location, route, lookahead=5.0):
    min_dist = float('inf')
    closest_command = RoadOption.LANEFOLLOW
    for wp, command in route:
        dist = ego_location.distance(wp.transform.location)
        if dist < min_dist and dist < lookahead:
            min_dist = dist
            closest_command = command
    return COMMAND_MAP.get(closest_command, -1)


# === ROS Node ===
class RoutePublisherNode(Node):
    def __init__(self, vehicle, route):
        super().__init__('route_publisher')
        self.vehicle = vehicle
        self.route = route

        self.cmd_pub = self.create_publisher(Int32, '/ego/high_level_command', 10)
        self.status_pub = self.create_publisher(CarlaEgoVehicleStatus, '/carla/ego_vehicle/vehicle_status', 10)

        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz

    def timer_callback(self):
        if not self.vehicle.is_alive:
            self.get_logger().warn("Ego vehicle no longer exists.")
            return

        loc = self.vehicle.get_location()
        velocity = self.vehicle.get_velocity()

        # Publish high-level command
        command_int = get_closest_command(loc, self.route)
        cmd_msg = Int32()
        cmd_msg.data = command_int
        self.cmd_pub.publish(cmd_msg)

        # Publish vehicle status (dummy or basic velocity only)
        status = CarlaEgoVehicleStatus()
        status.velocity = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        self.status_pub.publish(status)


# === Main Entry ===
def main():
    carla_proc = launch_carla()
    print("Launching CARLA...")
    time.sleep(10)

    client = carla.Client('172.28.129.33', 2000)
    client.set_timeout(10.0)
    world = client.load_world(MAP)
    blueprint_library = world.get_blueprint_library()

    # Setup Global Route Planner
    dao = GlobalRoutePlannerDAO(world.get_map(), SAMPLING_RESOLUTION)
    grp = GlobalRoutePlanner(dao)
    grp.setup()

    # Spawn ego vehicle
    spawn_point, end_point = random.choices(world.get_map().get_spawn_points(), k=2)
    vehicle_bp = blueprint_library.find(VEHICLE_TYPE)
    vehicle_bp.set_attribute("role_name", "ego")  # ROS 2 bridge uses this
    ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if not ego_vehicle:
        raise RuntimeError("Failed to spawn ego vehicle.")

    # Attach a camera
    sensor_bp = blueprint_library.find('sensor.camera.rgb')
    sensor_bp.set_attribute('image_size_x', '320')
    sensor_bp.set_attribute('image_size_y', '240')
    sensor_bp.set_attribute('fov', '90')
    sensor_bp.set_attribute('role_name', 'camera')

    camera_transform = carla.Transform(carla.Location(x=2.0, z=2.0), carla.Rotation(pitch=-10.0))
    camera = world.spawn_actor(sensor_bp, camera_transform, attach_to=ego_vehicle)
    camera.enable_for_ros()

    # Plan route
    start_wp = world.get_map().get_waypoint(ego_vehicle.get_location())
    end_wp = world.get_map().get_waypoint(end_point.location)
    route = grp.trace_route(start_wp.transform.location, end_wp.transform.location)

    if not route:
        raise RuntimeError("Route planning failed.")

    print("Route planned, starting ROS 2 node...")

    # Run ROS 2
    rclpy.init()
    node = RoutePublisherNode(ego_vehicle, route)
    rclpy.spin(node)

    # Cleanup
    node.destroy_node()
    rclpy.shutdown()
    camera.destroy()
    ego_vehicle.destroy()
    carla_proc.terminate()


if __name__ == '__main__':
    main()
