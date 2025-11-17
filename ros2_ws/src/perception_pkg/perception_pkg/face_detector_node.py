"""
Face Detector Node (Service-Based) - YuNet Model

DESCRIPTION:
    Face detection service using OpenCV's YuNet face detector.
    YuNet is a lightweight, accurate face detection model that provides
    clean outputs with reliable confidence scores.

RESPONSIBILITIES:
    1. Cache low-res camera frames
    2. Provide DetectFace service for on-demand face detection
    3. Extract ROI based on person track bounding box
    4. Run YuNet face detection on the ROI
    5. Return face detection results synchronously

MODEL DETAILS:
    - Model: YuNet (OpenCV's built-in face detector)
    - Input: RGB image (any size)
    - Output: Face bounding boxes with 5 landmarks and confidence scores
    - No complex preprocessing or postprocessing needed

SUBSCRIPTIONS:
    - /camera/image_low_res (sensor_msgs/Image): Low-res camera stream (640x360)

SERVICES:
    - /face_detection/detect_face (DetectFace): Detect face in person ROI

PARAMETERS:
    - confidence_threshold (float): Minimum face detection confidence [default: 0.5]
    - nms_threshold (float): NMS threshold for overlapping faces [default: 0.3]

COORDINATE SYSTEMS:
    - Input bbox: Low-res image coordinates (640x360)
    - ROI extraction: Crop from low-res image
    - YuNet output: Relative to ROI
    - Output bbox: Low-res image coordinates (640x360)

AUTHOR: Vittorio Albertin
DATE: 2025-10-29
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from msgs_interfaces.srv import DetectFace
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import urllib.request
import time

# YuNet model will be downloaded to models/yunet/ if not present


class FaceDetectorNode(Node):
    def __init__(self):
        super().__init__('face_detector_node')

        # Declare parameters
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.3)

        # Get parameters
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.nms_threshold = self.get_parameter('nms_threshold').value

        # Initialize YuNet face detector from local models directory
        try:
            # Use explicit path to models/yunet directory
            models_dir = '/ball-e/ros2_ws/models/yunet'
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'face_detection_yunet_2023mar.onnx')

            # Download if not present
            if not os.path.exists(model_path):
                self.get_logger().info(f"Downloading YuNet model to {model_path}...")
                model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                urllib.request.urlretrieve(model_url, model_path)
                self.get_logger().info("Model downloaded successfully")

            self.get_logger().info(f"YuNet model directory: {models_dir}")
            self.detector = cv2.FaceDetectorYN.create(
                model=model_path,
                config="",
                input_size=(320, 320),  # Will be updated dynamically
                score_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                top_k=5000
            )
            self.get_logger().info("YuNet face detector initialized successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize YuNet: {e}")
            raise RuntimeError(f"Failed to initialize YuNet: {e}")

        # Bridge for ROS-CV conversion
        self.bridge = CvBridge()

        # Cache for recent frames
        self.frame_cache = {}  # {timestamp_ns: Image}
        self.cache_duration_sec = 5.0  # Keep frames for 5 seconds (for enrollment)

        # Pinned frames cache (explicitly requested frames for identification)
        self.pinned_frames = {}  # {timestamp_ns: (Image, pin_time)}
        self.pinned_max_age = 30.0  # Auto-expire pinned frames after 30 seconds

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_low_res',
            self.image_callback,
            10
        )

        # Pin/release frame subscriptions
        self.pin_sub = self.create_subscription(
            Header,
            '/frame_cache/pin_timestamp',
            self.pin_frame_callback,
            10
        )
        self.release_sub = self.create_subscription(
            Header,
            '/frame_cache/release_timestamp',
            self.release_frame_callback,
            10
        )

        # Service server
        self.detect_face_service = self.create_service(
            DetectFace,
            '/face_detection/detect_face',
            self.detect_face_callback
        )

        # Cleanup timer
        self.create_timer(2, self.cleanup_cache)

        self.get_logger().info("FaceDetectorNode initialized (YuNet)")
        self.get_logger().info(f"  Confidence threshold: {self.conf_threshold}")
        self.get_logger().info(f"  NMS threshold: {self.nms_threshold}")

    def detect_face_callback(self, request, response):
        """
        Service callback for face detection.

        Args:
            request: DetectFace.Request containing track info and bbox
            response: DetectFace.Response to populate

        Returns:
            DetectFace.Response with face bbox or failure
        """
        # Convert timestamp to nanoseconds for cache key
        timestamp_ns = request.header.stamp.sec * 1_000_000_000 + request.header.stamp.nanosec

        self.get_logger().info(
            f"[DETECT] Track {request.track_id}: Timestamp={timestamp_ns}, " +
            f"Person bbox=({request.bbox_x}, {request.bbox_y}, {request.bbox_w}, {request.bbox_h})"
        )

        # Check if frame is available in pinned cache first, then rolling cache
        frame_msg = None
        if timestamp_ns in self.pinned_frames:
            frame_msg, _ = self.pinned_frames[timestamp_ns]
            self.get_logger().debug(f"[DETECT] Track {request.track_id}: Found frame in pinned cache")
        elif timestamp_ns in self.frame_cache:
            frame_msg = self.frame_cache[timestamp_ns]
            self.get_logger().debug(f"[DETECT] Track {request.track_id}: Found frame in rolling cache")
        else:
            response.success = False
            response.message = f"Frame not found in cache for timestamp {timestamp_ns}"
            self.get_logger().warning(f"[DETECT] Track {request.track_id}: {response.message}")
            return response

        try:
            # Convert ROS Image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(frame_msg, desired_encoding='bgr8')
            self.get_logger().info(f"[DETECT] Track {request.track_id}: Frame shape={frame.shape}")

            # Extract ROI
            x, y, w, h = request.bbox_x, request.bbox_y, request.bbox_w, request.bbox_h

            # Ensure ROI is within frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            if w <= 0 or h <= 0:
                response.success = False
                response.message = f"Invalid ROI for track {request.track_id}"
                self.get_logger().warning(f"[DETECT] Track {request.track_id}: {response.message}")
                return response

            roi = frame[y:y+h, x:x+w]
            self.get_logger().info(f"[DETECT] Track {request.track_id}: ROI extracted, shape={roi.shape}")

            # Update detector input size to match ROI
            self.detector.setInputSize((roi.shape[1], roi.shape[0]))

            # Detect faces in ROI
            _, faces = self.detector.detect(roi)

            if faces is None or len(faces) == 0:
                response.success = False
                response.message = f"No face detected for track {request.track_id}"
                self.get_logger().warning(f"[DETECT] Track {request.track_id}: {response.message}")
                return response

            self.get_logger().info(f"[DETECT] Track {request.track_id}: Found {len(faces)} faces")

            # YuNet output format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            # We only need: [x, y, w, h, score]

            # Log all detected faces
            for i, face in enumerate(faces):
                face_x, face_y, face_w, face_h = face[:4]
                confidence = face[14]
                self.get_logger().info(
                    f"[DETECT] Track {request.track_id}: Face {i}: " +
                    f"bbox=({face_x:.1f}, {face_y:.1f}, {face_w:.1f}, {face_h:.1f}), conf={confidence:.3f}"
                )

            # Take the most confident face
            best_face = max(faces, key=lambda f: f[14])
            face_x, face_y, face_w, face_h = best_face[:4]
            confidence = best_face[14]

            # Transform face bbox from ROI to full image coordinates
            full_face_x = x + face_x
            full_face_y = y + face_y

            self.get_logger().info(
                f"[DETECT] Track {request.track_id}: Best face transformed to full image: " +
                f"({full_face_x:.1f}, {full_face_y:.1f}, {face_w:.1f}, {face_h:.1f})"
            )

            # Populate response
            response.success = True
            response.face_x = int(full_face_x)
            response.face_y = int(full_face_y)
            response.face_w = int(face_w)
            response.face_h = int(face_h)
            response.message = f"Face detected (confidence: {confidence:.2f})"

            self.get_logger().info(
                f"[DETECT] Track {request.track_id}: SUCCESS - " +
                f"Face detected (confidence: {confidence:.2f})"
            )

            return response

        except Exception as e:
            response.success = False
            response.message = f"Face detection failed: {str(e)}"
            self.get_logger().error(f"[DETECT] Track {request.track_id}: ERROR - {response.message}")
            import traceback
            self.get_logger().error(f"[DETECT] Traceback: {traceback.format_exc()}")
            return response

    def image_callback(self, msg: Image):
        """
        Cache low-res camera frames.

        Args:
            msg: Image message from low-res camera stream
        """
        timestamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.frame_cache[timestamp_ns] = msg

    def pin_frame_callback(self, msg: Header):
        """
        Pin a specific frame timestamp for identification.

        Args:
            msg: Header with timestamp to pin
        """
        timestamp_ns = msg.stamp.sec * 1_000_000_000 + msg.stamp.nanosec

        # Check if frame is in rolling cache
        if timestamp_ns in self.frame_cache:
            self.pinned_frames[timestamp_ns] = (self.frame_cache[timestamp_ns], time.time())
            self.get_logger().info(f"[PIN] Pinned frame timestamp {timestamp_ns}")
        else:
            self.get_logger().warning(f"[PIN] Cannot pin timestamp {timestamp_ns} - not in rolling cache")

    def release_frame_callback(self, msg: Header):
        """
        Release a pinned frame timestamp.

        Args:
            msg: Header with timestamp to release
        """
        timestamp_ns = msg.stamp.sec * 1_000_000_000 + msg.stamp.nanosec

        if timestamp_ns in self.pinned_frames:
            del self.pinned_frames[timestamp_ns]
            self.get_logger().debug(f"[RELEASE] Released pinned frame {timestamp_ns}")

    def cleanup_cache(self):
        """Remove old frames from rolling cache and expired pinned frames."""
        now_ns = self.get_clock().now().nanoseconds
        cache_duration_ns = int(self.cache_duration_sec * 1e9)

        # Clean rolling cache
        old_frames = [ts for ts in self.frame_cache.keys() if now_ns - ts > cache_duration_ns]
        for ts in old_frames:
            del self.frame_cache[ts]

        # Clean expired pinned frames (safety net)
        now = time.time()
        expired_pins = [ts for ts, (_, pin_time) in self.pinned_frames.items()
                        if now - pin_time > self.pinned_max_age]
        for ts in expired_pins:
            del self.pinned_frames[ts]
            self.get_logger().warning(f"[CLEANUP] Auto-expired pinned frame {ts} after {self.pinned_max_age}s")


def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
