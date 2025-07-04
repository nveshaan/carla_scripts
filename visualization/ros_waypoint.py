import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
import pygame
import cv2

class WaypointVisualizer(Node):
    def __init__(self):
        super().__init__('waypoint_visualizer')
        self.bridge = CvBridge()

        # Image and waypoint storage
        self.current_image = None
        self.current_waypoints = []

        # ROS 2 Subscriptions
        self.create_subscription(Image, '/camera/image', self.image_callback, 10)
        self.create_subscription(Float32MultiArray, '/waypoints', self.waypoint_callback, 10)

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Waypoint Visualizer")
        self.running = True
        self.clock = pygame.time.Clock()

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
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
                surface = pygame.surfarray.make_surface(np.rot90(self.current_image))
                for (x, y) in self.current_waypoints:
                    pygame.draw.circle(surface, (0, 255, 0), (int(x), int(y)), 4)
                self.screen.blit(surface, (0, 0))

            pygame.display.flip()

        pygame.quit()

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