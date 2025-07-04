import carla
import subprocess
import time
import psutil
import random
import threading
import keyboard

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
        ["/mnt/western/Carla-0.10.0-Linux-Shipping/CarlaUnreal.sh", "--ros2", "-RenderOffScreen"],
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


# === ROS Nodes ===
class RoutePublisherNode(Node):
    def __init__(self, vehicle):
        super().__init__('velocity_publisher')
        self.vehicle = vehicle
        # self.route = route
        self.cmd = 4

        self.status_pub = self.create_publisher(CarlaEgoVehicleStatus, '/carla/ego/vehicle_status', 10)
        # self.cmd_pub = self.create_publisher(Int32, '/ego/high_level_command', 10)
        self.enable_autonomy = False # bool(route and len(route) > 0)

        if not self.enable_autonomy:
            self.get_logger().warn("Route not available. Disabling command publishing from planner.")

        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz

    def timer_callback(self):
        if not self.vehicle.is_alive:
            self.get_logger().warn("Ego vehicle no longer exists.")
            return

        loc = self.vehicle.get_location()
        velocity = self.vehicle.get_velocity()

        # Publish high-level command only if route is valid
        if self.enable_autonomy:
            command_int = get_closest_command(loc, self.route)
            if self.cmd != command_int:
                self.cmd = command_int
                self.get_logger().info(f"Publishing command: {command_int}")
                cmd_msg = Int32()
                cmd_msg.data = command_int
                self.cmd_pub.publish(cmd_msg)

        # Publish velocity
        status = CarlaEgoVehicleStatus()
        status.velocity = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        self.status_pub.publish(status)


class KeyboardCommandNode(Node):
    def __init__(self):
        super().__init__('keyboard_command_node')
        self.cmd_pub = self.create_publisher(Int32, '/ego/high_level_command', 10)
        self.running = True
        threading.Thread(target=self.keyboard_listener, daemon=True).start()

    def keyboard_listener(self):
        self.get_logger().info("Keyboard listener started. Press:")
        self.get_logger().info("  1 = LEFT, 2 = RIGHT, 3 = STRAIGHT, 4 = LANEFOLLOW")
        while self.running:
            if keyboard.is_pressed('1'):
                self.publish_cmd(RoadOption.LEFT)
            elif keyboard.is_pressed('2'):
                self.publish_cmd(RoadOption.RIGHT)
            elif keyboard.is_pressed('3'):
                self.publish_cmd(RoadOption.STRAIGHT)
            elif keyboard.is_pressed('4'):
                self.publish_cmd(RoadOption.LANEFOLLOW)
            time.sleep(0.1)

    def publish_cmd(self, road_option):
        msg = Int32()
        msg.data = COMMAND_MAP[road_option]
        self.cmd_pub.publish(msg)
        self.get_logger().info(f"Published manual command: {msg.data}")


# === Main Entry ===
def main():
    # carla_proc = launch_carla()
    # print("Launching CARLA...")
    # time.sleep(10)

    client = carla.Client('localhost', 1946)
    client.set_timeout(10.0)
    world = client.load_world(MAP)
    blueprint_library = world.get_blueprint_library()

    # Setup Global Route Planner
    # dao = GlobalRoutePlannerDAO(world.get_map(), SAMPLING_RESOLUTION)
    # grp = GlobalRoutePlanner(dao)
    # grp.setup()

    # Spawn ego vehicle
    spawn_point, end_point = random.choices(world.get_map().get_spawn_points(), k=2)
    vehicle_bp = blueprint_library.find(VEHICLE_TYPE)
    vehicle_bp.set_attribute("ros_name", "ego")
    ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    # ego_vehicle.enable_for_ros()
    if not ego_vehicle:
        raise RuntimeError("Failed to spawn ego vehicle.")

    # Attach a camera
    sensor_bp = blueprint_library.find('sensor.camera.rgb')
    sensor_bp.set_attribute('image_size_x', '320')
    sensor_bp.set_attribute('image_size_y', '240')
    sensor_bp.set_attribute('fov', '90')
    sensor_bp.set_attribute('ros_name', 'camera')

    camera_transform = carla.Transform(carla.Location(x=2.0, z=2.0), carla.Rotation(pitch=-10.0))
    camera = world.spawn_actor(sensor_bp, camera_transform, attach_to=ego_vehicle)
    camera.enable_for_ros()

    # Try planning route
    # start_wp = world.get_map().get_waypoint(ego_vehicle.get_location())
    # end_wp = world.get_map().get_waypoint(end_point.location)
    # route = grp.trace_route(start_wp.transform.location, end_wp.transform.location)

    rclpy.init()

    try:
        route_node = RoutePublisherNode(ego_vehicle)
      #  if not route_node.enable_autonomy:
      #      keyboard_node = KeyboardCommandNode()
      #      executor = rclpy.executors.MultiThreadedExecutor()
      #      executor.add_node(route_node)
      #      executor.add_node(keyboard_node)
      #      executor.spin()
      #      keyboard_node.destroy_node()
      #  else:
        rclpy.spin(route_node)

        route_node.destroy_node()

    finally:
        rclpy.shutdown()
        camera.destroy()
        ego_vehicle.destroy()
        # carla_proc.terminate()


if __name__ == '__main__':
    main()
