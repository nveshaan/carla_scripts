import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
import pygame
import cv2

# === Coordinate Transformation ===

def ego_to_camera(points):
    x, y = points[:, 0], points[:, 1]
    z = np.zeros_like(x)
    cam = np.stack([y, -z, x], axis=1)  # [right, down, forward]
    cam += np.array([0.0, 2.0, 2.0])    # camera offset (2m up, 2m forward)

    pitch = np.radians(10.0)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])
    return (Rx @ cam.T).T

def project_to_image(cam_pts, width=320, height=240, fov=90.0):
    fx = fy = width / (2 * np.tan(np.radians(fov / 2)))
    cx, cy = width / 2, height / 2
    x, y, z = cam_pts[:, 0], cam_pts[:, 1], np.clip(cam_pts[:, 2], 1e-5, None)
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return np.stack([u, v], axis=1)

# === Main Node ===

class WaypointVisualizer(Node):
    def __init__(self):
        super().__init__('waypoint_visualizer')
        self.bridge = CvBridge()

        # Display settings
        self.img_w, self.img_h = 320, 240
        self.scale = 2

        self.current_image = None
        self.current_waypoints = []

        pygame.init()
        self.screen = pygame.display.set_mode((self.img_w * self.scale, self.img_h * self.scale))
        pygame.display.set_caption("Waypoint Visualizer")
        self.clock = pygame.time.Clock()
        self.running = True

        self.create_subscription(Image, '/camera/image', self.image_callback, 10)
        self.create_subscription(Float32MultiArray, '/waypoints', self.waypoint_callback, 10)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def waypoint_callback(self, msg):
        data = msg.data
        if len(data) % 2 == 0:
            self.current_waypoints = list(zip(data[::2], data[1::2]))

    def run(self):
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.01)
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            if self.current_image is not None:
                # Resize and rotate image to match pygame layout
                resized = cv2.resize(self.current_image, (self.img_w * self.scale, self.img_h * self.scale))
                rotated = np.rot90(resized)
                surface = pygame.surfarray.make_surface(rotated.swapaxes(0, 1))

                # Project and draw waypoints
                if self.current_waypoints:
                    pts = np.array(self.current_waypoints)
                    cam_pts = ego_to_camera(pts)
                    img_pts = project_to_image(cam_pts, self.img_w, self.img_h)
                    for pt in img_pts:
                        u, v = int(pt[0] * self.scale), int(pt[1] * self.scale)
                        if 0 <= u < self.img_w * self.scale and 0 <= v < self.img_h * self.scale:
                            pygame.draw.circle(surface, (0, 255, 0), (u, v), 4)

                self.screen.blit(surface, (0, 0))
                pygame.display.flip()

        pygame.quit()

# === Entry Point ===

def main():
    rclpy.init()
    node = WaypointVisualizer()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()