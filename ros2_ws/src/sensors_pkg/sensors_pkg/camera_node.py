import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Publisher
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)

        # OpenCV capture (0 = default laptop camera)
        self.cap = cv2.VideoCapture(0)

        # Bridge between cv2 and ROS images
        self.bridge = CvBridge()

        # Timer at ~30Hz (adjust if needed)
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)

        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera!")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame")
            return

        # Convert BGR (OpenCV default) → ROS Image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)

    def destroy_node(self):
        # Release camera when shutting down
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
