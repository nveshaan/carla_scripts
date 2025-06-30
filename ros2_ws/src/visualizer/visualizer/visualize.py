import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray, Marker

class WaypointVisualizerNode(Node):
    def __init__(self):
        super().__init__('waypoint_visualizer_node')

        self.create_subscription(Float32MultiArray, '/waypoint_pid/debug_waypoints', self.callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoint_pid/visualization', 10)

    def callback(self, msg):
        data = msg.data
        if len(data) % 2 != 0:
            return

        points = list(zip(data[::2], data[1::2]))

        marker_array = MarkerArray()
        for i, (x, y) in enumerate(points):
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "waypoints"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.2
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 1.0
            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = WaypointVisualizerNode()
    rclpy.spin(node)
    rclpy.shutdown()
