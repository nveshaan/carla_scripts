import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray, Int32
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus

import time
import numpy as np
from collections import deque

# ===================== Utility Functions =====================
def ls_circle(points):
    xs = points[:, 0]
    ys = points[:, 1]

    us = xs - np.mean(xs)
    vs = ys - np.mean(ys)

    Suu = np.sum(us ** 2)
    Suv = np.sum(us * vs)
    Svv = np.sum(vs ** 2)
    Suuu = np.sum(us ** 3)
    Suvv = np.sum(us * vs * vs)
    Svvv = np.sum(vs ** 3)
    Svuu = np.sum(vs * us * us)

    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = np.array([0.5 * Suuu + 0.5 * Suvv, 0.5 * Svvv + 0.5 * Svuu])

    cx, cy = np.linalg.solve(A, b)
    r = np.sqrt(cx * cx + cy * cy + (Suu + Svv) / len(xs))

    cx += np.mean(xs)
    cy += np.mean(ys)

    return np.array([cx, cy]), r


class PIDController:
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, fps=20, n=30):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._dt = 1.0 / fps
        self._window = deque(maxlen=n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = sum(self._window) * self._dt
            derivative = (self._window[-1] - self._window[-2]) / self._dt
        else:
            integral = 0.0
            derivative = 0.0

        control = self._K_P * error + self._K_I * integral + self._K_D * derivative
        return control

    def reset(self):
        self._window.clear()


class CustomController:
    def __init__(self, controller_args, dt=0.05):
        self._controller_args = controller_args
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

    def run_step(self, alpha, cmd):
        self._e_buffer.append(alpha)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        args = self._controller_args.get(str(cmd), self._controller_args["3"])
        return args["Kp"] * alpha + args["Kd"] * _de + args["Ki"] * _ie

    def reset(self):
        self._e_buffer.clear()


# ===================== Main ROS 2 Node =====================
class WaypointPIDController(Node):
    def __init__(self):
        super().__init__('waypoint_pid_controller')

        self.fps = 20
        self.dt = 1.0 / self.fps
        self.gap = 5
        self.command = 4  # Default command

        self.steer_points = {"1": 4, "2": 3, "3": 2, "4": 2}
        self.pid_gains = {
            "1": {"Kp": 0.5, "Ki": 0.20, "Kd": 0.0},
            "2": {"Kp": 0.7, "Ki": 0.10, "Kd": 0.0},
            "3": {"Kp": 1.0, "Ki": 0.10, "Kd": 0.0},
            "4": {"Kp": 1.0, "Ki": 0.50, "Kd": 0.0},
        }

        self.turn_control = CustomController(self.pid_gains, dt=self.dt)
        self.speed_control = PIDController(K_P=0.5, K_I=0.08, K_D=0.0, fps=self.fps)

        self.current_speed = 10.0
        self.waypoints = None

        self.create_subscription(Float32MultiArray, '/prediction/waypoints', self.waypoint_callback, 10)
        self.create_subscription(CarlaEgoVehicleStatus, '/carla/ego/vehicle_status', self.status_callback, 10)
        # self.create_subscription(Int32, '/carla/ego/high_level_command', self.command_callback, 10)

        self.control_pub = self.create_publisher(CarlaEgoVehicleControl, '/carla/ego/vehicle_control_cmd', 10)
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.debug_pub = self.create_publisher(Float32MultiArray, '/waypoint_pid/debug_waypoints', 10)

        start_time = time.time()
        while time.time() - start_time < 60:
            _ctrl = CarlaEgoVehicleControl()
            _ctrl.throttle = float(0.6)
            _ctrl.steer = float(0)
            _ctrl.brake = float(0)
            _ctrl.reverse = False
            _ctrl.hand_brake = False
            self.control_pub.publish(_ctrl)
            time.sleep(1)

    def waypoint_callback(self, msg):
        data = np.array(msg.data)
        if data.size != 10:
            self.get_logger().warn("Expected 5 waypoints, got incorrect size.")
            return
        self.waypoints = data.reshape(5, 2)

        # Reset controllers on new waypoints
        self.turn_control.reset()
        self.speed_control.reset()

    def status_callback(self, msg):
        self.current_speed = msg.velocity

    def command_callback(self, msg):
        self.command = msg.data

    def control_loop(self):
        if self.waypoints is None or len(self.waypoints) < 2:
            return

        STEPS = min(len(self.waypoints), 5)
        targets = [(0.0, 0.0)]

        for i in range(STEPS):
            dx, dy = self.waypoints[i]
            angle = np.arctan2(dx, dy)
            dist = np.linalg.norm([dx, dy])
            targets.append([dist * np.cos(angle), dist * np.sin(angle)])
        targets = np.array(targets)

        debug_msg = Float32MultiArray()
        debug_msg.data = targets.flatten().tolist()
        self.debug_pub.publish(debug_msg)

        c, r = ls_circle(targets)
        n = self.steer_points.get(str(self.command), 1)
        target_pt = targets[n]
        vec = target_pt - c
        vec = vec / np.linalg.norm(vec) * r
        closest = c + vec

        v = np.array([1.0, 0.0])
        w = closest / np.linalg.norm(closest)
        alpha = np.arctan2(w[1], w[0])

        steer = np.clip(self.turn_control.run_step(alpha, self.command), -1.0, 1.0)

        dists = np.linalg.norm(self.waypoints[:-1] - self.waypoints[1:], axis=1)
        target_speed = dists.mean() / (self.gap * self.dt) if len(dists) > 0 else 0.0
        acceleration = target_speed - self.current_speed
        throttle = np.clip(self.speed_control.step(acceleration), 0.0, 1.0)

        brake = 0.0
        if target_speed <= 2.0:
            steer = 0.0
            throttle = 0.0
        if target_speed <= 1.0:
            brake = 1.0

        ctrl = CarlaEgoVehicleControl()
        ctrl.throttle = float(throttle)
        ctrl.steer = float(steer)
        ctrl.brake = float(brake)
        ctrl.reverse = False
        ctrl.hand_brake = False
        self.control_pub.publish(ctrl)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointPIDController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
