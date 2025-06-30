import torch
import torchvision.transforms as T
from torch_inference.models.image_net import ImagePolicyModel

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Float32
from carla_msgs.msg import CarlaEgoVehicleStatus

from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

class WaypointPredictionNode(Node):
    """
    Subscription: /image, /velocity, /command
    Publication: /waypoints
    """
    def __init__(self):
        super().__init__("inference_node")
        self.declare_parameter("env", "sim")
        self.declare_parameter("wait", 20.0)
        self.env = self.get_parameter("env").get_parameter_value().string_value
        self.wait = self.get_parameter("wait").get_parameter_value().double_value

        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_dir, "checkpoints", "0627_1556_model.pth")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = T.Compose([
            T.Lambda(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
        ])
        
        self.latest_img = None
        self.latest_vel = None
        self.latest_cmd = None

        if self.env == "sim":
            self.img_sub = self.create_subscription(Image, '/carla/ego/camera/image', self.img_callback, 10)
            self.vel_sub = self.create_subscription(CarlaEgoVehicleStatus, 'carla/ego/vehicle_status', self.vel_callback, 10)
            self.cmd_sub = self.create_subscription(Float32, 'carla/ego/high_command', self.cmd_callback, 10)
        else:
            pass # TODO: check the real topic names

        self.timer = self.create_timer(self.wait, self.update_waypoints)
        self.publisher_ = self.create_publisher(Float32MultiArray, '/prediction/waypoints', 10)
        self.bridge = CvBridge()
        self.model = ImagePolicyModel(backbone="resnet34")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True), strict=False)
        self.model.eval()

    def img_callback(self, img):
        self.latest_img = img

    def vel_callback(self, vel):
        self.latest_vel = vel

    def cmd_callback(self, cmd):
        self.latest_cmd = cmd
        self.update_waypoints()
        
    def update_waypoints(self):
        if self.latest_img is None:
            self.get_logger().warn("No data received yet.")
            return
        if self.latest_cmd is None:
            self.latest_cmd = 4.0
        if self.latest_vel is None:
            self.latest_vel = 0.0
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_img, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error('Failed to convert image: %s' % str(e))
            return
        
        try:
            image_tensor = self.transform(cv_image)
            waypoints = self.model(torch.tensor([image_tensor], torch.float32, self.device),
                                   torch.tensor([self.latest_vel], torch.float32, self.device),
                                   torch.tensor([self.latest_cmd], torch.float32, self.device))[0].cpu().numpy()
        except RuntimeError as e:
            self.get_logger().error('Failed to infer model: %s' % str(e))
            return

        try:
            self.publisher_.publish(waypoints)
        except CvBridgeError as e:
            self.get_logger().error('Failed to publish waypoints: %s' % str(e))

def main(args=None):
    rclpy.init(args=args)
    node = WaypointPredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
