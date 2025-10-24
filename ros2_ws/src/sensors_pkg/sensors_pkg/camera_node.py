import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Declare parameters
        self.declare_parameter('camera_index', 2)  # Default to USB camera (index 2)
        self.declare_parameter('fps', 30.0)  # Configurable frame rate
        self.declare_parameter('width', 1920)  # 4K width (3840x2160)
        self.declare_parameter('height', 1080)  # 4K height

        camera_index = self.get_parameter('camera_index').value
        fps = self.get_parameter('fps').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value

        # Publisher
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)

        # OpenCV capture
        # Camera 0 = HP laptop camera (/dev/video0)
        # Camera 2 = UVC USB camera (/dev/video2)
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open camera {camera_index}!")
            self.get_logger().error("Available cameras: 0 (laptop), 2 (USB)")
            return

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Verify actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Bridge between cv2 and ROS images
        self.bridge = CvBridge()

        # Timer at configured FPS
        self.timer = self.create_timer(1.0/fps, self.timer_callback)

        # Log camera configuration
        self.get_logger().info(f"Camera {camera_index} opened successfully")
        self.get_logger().info(f"Requested: {width}x{height} @ {fps} fps")
        self.get_logger().info(f"Actual:    {actual_width}x{actual_height} @ {actual_fps} fps")

        if actual_width != width or actual_height != height:
            self.get_logger().warning(
                f"Camera resolution mismatch! Requested {width}x{height}, got {actual_width}x{actual_height}"
            )
            self.get_logger().warning("Try: 3840x2160, 2560x1440, 1920x1080, 1280x720, or 640x480")

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
