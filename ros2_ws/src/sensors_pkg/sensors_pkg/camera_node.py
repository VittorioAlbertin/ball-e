import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Declare parameters
        self.declare_parameter('camera_index', 0)  # Default to laptop camera (index 0)
        self.declare_parameter('fps', 30.0)  # Configurable frame rate
        self.declare_parameter('width', 1920)  # High-res width
        self.declare_parameter('height', 1080)  # High-res height
        self.declare_parameter('low_res_width', 640)  # Low-res width for YOLO
        self.declare_parameter('low_res_height', 360)  # Low-res height for YOLO
        self.declare_parameter('publish_fps', 30.0)  # Publish rate for both streams

        camera_index = self.get_parameter('camera_index').value
        fps = self.get_parameter('fps').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        low_res_width = self.get_parameter('low_res_width').value
        low_res_height = self.get_parameter('low_res_height').value
        publish_fps = self.get_parameter('publish_fps').value

        # Store low-res dimensions for callbacks
        self.low_res_width = low_res_width
        self.low_res_height = low_res_height

        # Dual-stream publishers
        self.publisher_raw = self.create_publisher(Image, '/camera/image_raw', 10)
        self.publisher_low_res = self.create_publisher(Image, '/camera/image_low_res', 10)

        # OpenCV capture
        # Camera 0 = HP laptop camera (/dev/video0)
        # Camera 2 = UVC USB camera (/dev/video2)
        # sometimes they swap, just check manually with `v4l2 --list-devices` on a terminal
        # Use V4L2 backend explicitly for better control on Linux
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open camera {camera_index}!")
            return

        # Set resolution and FPS
        # Note: cap.set() returning True doesn't guarantee the value was accepted
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency with small buffer
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Verify actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Bridge between cv2 and ROS images
        self.bridge = CvBridge()

        # Single timer for synchronized dual-stream publishing
        # Both streams published at same rate with same timestamp
        self.publish_timer = self.create_timer(1.0/publish_fps, self.publish_callback)

        # Log camera configuration
        self.get_logger().info(f"Camera {camera_index} opened successfully")
        self.get_logger().info(f"Requested: {width}x{height} @ {fps} fps")
        self.get_logger().info(f"Actual:    {actual_width}x{actual_height} @ {actual_fps} fps")
        self.get_logger().info(f"Dual-stream mode (synchronized):")
        self.get_logger().info(f"  High-res: {width}x{height} @ {publish_fps} Hz → /camera/image_raw")
        self.get_logger().info(f"  Low-res:  {low_res_width}x{low_res_height} @ {publish_fps} Hz → /camera/image_low_res")
        self.get_logger().info(f"  Both streams share identical timestamps")

        if actual_width != width or actual_height != height:
            self.get_logger().warning(
                f"Camera resolution mismatch! Requested {width}x{height}, got {actual_width}x{actual_height}"
            )
            self.get_logger().warning("Try: 3840x2160, 2560x1440, 1920x1080, 1280x720, or 640x480")

    def publish_callback(self):
        """Capture and publish both high-res and low-res frames with same timestamp."""
        # Capture frame from camera
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn("Failed to capture frame", throttle_duration_sec=5.0)
            return

        try:
            # Get current timestamp - this will be shared by both streams
            timestamp = self.get_clock().now().to_msg()

            # Publish high-resolution frame
            high_res_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            high_res_msg.header.stamp = timestamp
            high_res_msg.header.frame_id = 'camera_high_res'
            self.publisher_raw.publish(high_res_msg)

            # Resize to low resolution
            low_res_frame = cv2.resize(frame, (self.low_res_width, self.low_res_height))

            # Publish low-resolution frame with SAME timestamp
            low_res_msg = self.bridge.cv2_to_imgmsg(low_res_frame, encoding='bgr8')
            low_res_msg.header.stamp = timestamp
            low_res_msg.header.frame_id = 'camera_low_res'
            self.publisher_low_res.publish(low_res_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to publish frames: {e}", throttle_duration_sec=5.0)

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
