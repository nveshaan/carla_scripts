import torch
import torchvision.transforms as T
from torch_inference.models.image_net import ImagePolicyModel

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

class WaypointPredictionNode(Node):
    def __init__(self):
        super().__init__("inference_node")
        self.declare_parameter("env", "sim")
        self.declare_parameter("wait", 10.0)
        self.env = self.get_parameter("env").get_parameter_value().string_value
        self.wait = self.get_parameter("wait").get_parameter_value().double_value

        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_dir, "checkpoints", "0627_1556_model.pth")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = T.Compose([
            T.Lambda(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
            T.ToTensor()
        ])
        
        self.latest_img = None
        if self.env == "sim":
            self.subscription = self.create_subscription(Image, '/carla/ego/front_camera/image', self.listener_callback, 10)
        else:
            pass # TODO: check the real image topic

        self.timer = self.create_timer(self.wait, self.timer_callback)
        self.publisher_ = self.create_publisher(Float32MultiArray, '/prediction/waypoints', 10)
        self.bridge = CvBridge()
        self.model = ImagePolicyModel(backbone="resnet34")
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
        self.model.eval()

    def listener_callback(self, img):
        self.latest_img = img
        
    def timer_callback(self):
        if self.latest_img is None:
            self.get_logger().warn("No image received yet.")
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_img, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error('Failed to convert image: %s' % str(e))
            return
        
        image_tensor = self.transform(cv_image)
        waypoints = self.model([image_tensor])[0]

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
