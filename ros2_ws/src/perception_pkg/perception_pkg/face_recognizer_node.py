"""
Face Recognizer Node

DESCRIPTION:
    The FaceRecognizer node is responsible for generating face embeddings using the
    FaceNet model and matching them against the people database. It receives face
    detections with bounding boxes, extracts the face crop from the high-resolution
    camera stream, generates a 512-dimensional embedding vector, and calls the
    recognition service to identify the person.

RESPONSIBILITIES:
    1. Subscribe to face detections from FaceDetectorNode
    2. Synchronize with high-res camera frames using timestamps
    3. Scale face bbox from low-res (640x360) to high-res (1920x1080) coordinates
    4. Extract and preprocess face crops from high-res images
    5. Generate face embeddings using FaceNet (facenet.onnx)
    6. Call recognize_face service to match against database
    7. Publish identity results to PersonStateManager

MODEL DETAILS:
    - Model: FaceNet (facenet.onnx)
    - Input: 160x160x3 RGB image, normalized [-1, 1]
    - Output: 512-dimensional embedding vector (L2 normalized)
    - Recognition: Cosine similarity matching with threshold

SUBSCRIPTIONS:
    - /camera/image_raw (sensor_msgs/Image): High-res camera stream (1920x1080)
    - /face_detector/detections (FaceRecognition): Face bounding boxes with track IDs

PUBLICATIONS:
    - /face_recognizer/results (IdentityUpdate): Identity results with confidence

SERVICE CLIENTS:
    - people_db/recognize_face: Face recognition matching service

PARAMETERS:
    - model_path (str): Path to facenet.onnx model [default: models/facenet.onnx]
    - recognition_threshold (float): Minimum cosine similarity for match [default: 0.6]
    - use_gpu (bool): Use GPU acceleration if available [default: true]
    - low_res_width (int): Low-res image width for bbox scaling [default: 640]
    - low_res_height (int): Low-res image height for bbox scaling [default: 360]
    - high_res_width (int): High-res image width for bbox scaling [default: 1920]
    - high_res_height (int): High-res image height for bbox scaling [default: 1080]

DATA FLOW:
    1. Cache high-res frames continuously
    2. Receive face detection (track_id, bbox in low-res, timestamp)
    3. Find high-res frame with matching timestamp
    4. Scale bbox from low-res (640x360) to high-res (1920x1080)
    5. Crop face from high-res image
    6. Preprocess: Resize to 160x160, normalize [-1, 1]
    7. Generate 512-dim embedding with FaceNet
    8. Call recognize_face service
    9. Publish identity result

COORDINATE SCALING:
    Input bbox is in low-res coordinates (640x360), must be scaled to high-res (1920x1080):
    - scale_x = 1920 / 640 = 3.0
    - scale_y = 1080 / 360 = 3.0
    - high_res_x = low_res_x * scale_x
    - high_res_w = low_res_w * scale_x

GPU ACCELERATION:
    - Attempts to use CUDAExecutionProvider if available
    - Falls back to CPUExecutionProvider if GPU not available
    - Check available providers: onnxruntime.get_available_providers()

AUTHOR: Vittorio Albertin
DATE: 2025-10-29
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from msgs_interfaces.msg import FaceRecognition, IdentityUpdate
from msgs_interfaces.srv import RecognizeFace
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
import os
import warnings

# Suppress ONNX Runtime warnings
warnings.filterwarnings('ignore', category=UserWarning, module='onnxruntime')
ort.set_default_logger_severity(3)  # 3 = ERROR level (suppress warnings)

FACENET_MODEL_PATH = "/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/facenet.onnx"


class FaceRecognizerNode(Node):
    def __init__(self):
        super().__init__('face_recognizer_node')

        # Declare parameters
        self.declare_parameter('recognition_threshold', 0.6)
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('low_res_width', 640)
        self.declare_parameter('low_res_height', 360)
        self.declare_parameter('high_res_width', 1920)
        self.declare_parameter('high_res_height', 1080)
        self.declare_parameter('face_crops_dir', '/ball-e/ros2_ws/robot_data/face_crops')

        # Get parameters
        self.recognition_threshold = self.get_parameter('recognition_threshold').value
        use_gpu = self.get_parameter('use_gpu').value
        self.low_res_width = self.get_parameter('low_res_width').value
        self.low_res_height = self.get_parameter('low_res_height').value
        self.high_res_width = self.get_parameter('high_res_width').value
        self.high_res_height = self.get_parameter('high_res_height').value
        self.face_crops_dir = self.get_parameter('face_crops_dir').value

        # Compute scaling factors
        self.scale_x = self.high_res_width / self.low_res_width
        self.scale_y = self.high_res_height / self.low_res_height

        # Create face crops directory if it doesn't exist
        os.makedirs(self.face_crops_dir, exist_ok=True)
        self.get_logger().info(f"Face crops will be saved to: {self.face_crops_dir}")

        # Check if model exists
        if not os.path.exists(FACENET_MODEL_PATH):
            self.get_logger().error(f"Model file not found: {FACENET_MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {FACENET_MODEL_PATH}")

        # Setup execution providers (GPU/CPU)
        providers = ['CPUExecutionProvider']  # Default to CPU

        if use_gpu:
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                try:
                    # Try to create session with CUDA first
                    test_session = ort.InferenceSession(
                        FACENET_MODEL_PATH,
                        providers=['CUDAExecutionProvider']
                    )
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self.get_logger().info("GPU acceleration enabled (CUDA)")
                except Exception:
                    # CUDA libraries not available, fall back to CPU
                    self.get_logger().info("CUDA not available, using CPU")
            elif 'TensorrtExecutionProvider' in available_providers:
                providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
                self.get_logger().info("GPU acceleration enabled (TensorRT)")
            else:
                self.get_logger().info("GPU not available, using CPU")
        else:
            self.get_logger().info("GPU disabled, using CPU")

        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(
            FACENET_MODEL_PATH,
            providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Get model input shape
        input_shape = self.session.get_inputs()[0].shape
        self.model_input_size = input_shape[2]  # Assuming square input [batch, channels, height, width]

        # Log active provider
        active_provider = self.session.get_providers()[0]
        self.get_logger().info(f"Using execution provider: {active_provider}")
        self.get_logger().info(f"Model input size: {self.model_input_size}x{self.model_input_size}")

        # Bridge for ROS-CV conversion
        self.bridge = CvBridge()

        # Cache for high-res frames
        self.frame_cache = {}  # {timestamp_ns: Image}
        self.cache_duration_sec = 1.0  # Keep frames for 1 second

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            FaceRecognition,
            '/face_detector/detections',
            self.detection_callback,
            10
        )

        # Publications
        self.result_pub = self.create_publisher(IdentityUpdate, '/face_recognizer/results', 10)

        # Service client
        self.recognize_client = self.create_client(RecognizeFace, 'people_db/recognize_face')

        # Cleanup timer
        self.create_timer(0.5, self.cleanup_cache)

        self.get_logger().info("FaceRecognizerNode initialized")
        self.get_logger().info(f"  Model: {FACENET_MODEL_PATH}")
        self.get_logger().info(f"  Recognition threshold: {self.recognition_threshold}")
        self.get_logger().info(f"  Resolution scaling: {self.low_res_width}x{self.low_res_height} → {self.high_res_width}x{self.high_res_height}")
        self.get_logger().info(f"  Scale factors: x={self.scale_x:.2f}, y={self.scale_y:.2f}")

    def image_callback(self, msg: Image):
        """
        Cache high-res camera frames for face extraction.

        Args:
            msg: Image message from high-res camera stream
        """
        timestamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.frame_cache[timestamp_ns] = msg

    def detection_callback(self, msg: FaceRecognition):
        """
        Process face detection and perform recognition.

        Args:
            msg: FaceRecognition message containing face bbox and track ID
        """
        timestamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        # Check if high-res frame is available
        if timestamp_ns not in self.frame_cache:
            self.get_logger().warning(
                f"High-res frame not found for timestamp {timestamp_ns} " +
                f"(track {msg.track_id}). Skipping recognition."
            )
            return

        try:
            # Get high-res frame
            frame_msg = self.frame_cache[timestamp_ns]
            frame = self.bridge.imgmsg_to_cv2(frame_msg, desired_encoding='bgr8')

            # Scale bbox from low-res to high-res
            high_res_bbox = self.scale_bbox(
                msg.face_x, msg.face_y, msg.face_w, msg.face_h
            )

            self.get_logger().info(
                f"Track {msg.track_id} - Low-res bbox: ({msg.face_x}, {msg.face_y}, {msg.face_w}, {msg.face_h})"
            )
            self.get_logger().info(
                f"Track {msg.track_id} - High-res bbox: {high_res_bbox}, Frame shape: {frame.shape}"
            )

            # Extract face crop
            face_crop = self.extract_face(frame, high_res_bbox)

            if face_crop is None:
                self.get_logger().warning(
                    f"Failed to extract face for track {msg.track_id}. " +
                    f"Bbox: {high_res_bbox}, Frame: {frame.shape}"
                )
                # Publish unknown result
                result = IdentityUpdate()
                result.header = msg.header
                result.track_id = msg.track_id
                result.identity = 'unknown'
                result.confidence = 0.0
                result.person_id = -1
                self.result_pub.publish(result)
                return

            # Save high-res face crop to disk
            self.save_face_crop(face_crop, msg.track_id, timestamp_ns)

            # Generate embedding
            embedding = self.generate_embedding(face_crop)

            # Recognize face
            identity, confidence, person_id = self.recognize_face(embedding)

            # Publish result
            result = IdentityUpdate()
            result.header = msg.header
            result.track_id = msg.track_id
            result.identity = identity
            result.confidence = confidence
            result.person_id = person_id

            self.result_pub.publish(result)

            self.get_logger().info(
                f"Recognition result for track {msg.track_id}: " +
                f"'{identity}' (confidence: {confidence:.3f}, person_id: {person_id})"
            )

        except Exception as e:
            self.get_logger().error(f"Face recognition failed for track {msg.track_id}: {e}")

    def scale_bbox(self, x: int, y: int, w: int, h: int) -> tuple:
        """
        Scale bounding box from low-res to high-res coordinates.

        Args:
            x, y, w, h: Bbox in low-res coordinates (640x360)

        Returns:
            tuple: (x, y, w, h) in high-res coordinates (1920x1080)
        """
        high_x = int(x * self.scale_x)
        high_y = int(y * self.scale_y)
        high_w = int(w * self.scale_x)
        high_h = int(h * self.scale_y)

        return (high_x, high_y, high_w, high_h)

    def extract_face(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Extract face crop from frame with bounds checking.

        Args:
            frame: High-res frame (1920x1080x3)
            bbox: (x, y, w, h) in high-res coordinates

        Returns:
            np.ndarray: Face crop or None if invalid
        """
        x, y, w, h = bbox

        # Validate input bbox
        if w <= 0 or h <= 0:
            self.get_logger().debug(f"Invalid input bbox: w={w}, h={h}")
            return None

        # Add margin (30%) to include more context
        margin_x = int(w * 0.3)
        margin_y = int(h * 0.3)

        # Apply margins
        x_start = max(0, x - margin_x)
        y_start = max(0, y - margin_y)
        x_end = min(frame.shape[1], x + w + margin_x)
        y_end = min(frame.shape[0], y + h + margin_y)

        # Check if valid crop region
        crop_w = x_end - x_start
        crop_h = y_end - y_start

        if crop_w <= 0 or crop_h <= 0:
            self.get_logger().debug(
                f"Invalid crop region: x_start={x_start}, y_start={y_start}, " +
                f"x_end={x_end}, y_end={y_end}, crop_w={crop_w}, crop_h={crop_h}"
            )
            return None

        crop = frame[y_start:y_end, x_start:x_end]
        return crop

    def generate_embedding(self, face: np.ndarray) -> np.ndarray:
        """
        Generate 512-dim face embedding using FaceNet.

        Args:
            face: Face crop (BGR)

        Returns:
            np.ndarray: 512-dim embedding vector (L2 normalized)
        """
        # Preprocess
        preprocessed = self.preprocess_face(face)

        # Inference
        embedding = self.session.run([self.output_name], {self.input_name: preprocessed})[0]

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.flatten()

    def preprocess_face(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess face for FaceNet model.

        Args:
            face: Face crop (BGR, HxWx3)

        Returns:
            np.ndarray: Preprocessed tensor (1x3xHxW) where H,W = model_input_size
        """
        # Resize to model input size (112x112 or 160x160 depending on model)
        resized = cv2.resize(face, (self.model_input_size, self.model_input_size))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1]
        normalized = (rgb.astype(np.float32) - 127.5) / 128.0

        # Transpose to CHW format
        chw = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batch = np.expand_dims(chw, axis=0)

        return batch

    def recognize_face(self, embedding: np.ndarray) -> tuple:
        """
        Call recognition service to match embedding against database.

        Args:
            embedding: 512-dim face embedding

        Returns:
            tuple: (identity, confidence, person_id)
        """
        # Wait for service
        if not self.recognize_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("recognize_face service not available")
            return ('unknown', 0.0, -1)

        # Create request
        request = RecognizeFace.Request()
        request.face_embedding = embedding.tolist()
        request.confidence_threshold = self.recognition_threshold

        try:
            # Call service synchronously
            future = self.recognize_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

            if future.done():
                response = future.result()
                if response.match_found:
                    return (response.person_name, response.similarity_score, response.person_id)
                else:
                    return ('unknown', 0.0, -1)
            else:
                self.get_logger().error("recognize_face service call timeout")
                return ('unknown', 0.0, -1)

        except Exception as e:
            self.get_logger().error(f"recognize_face service call failed: {e}")
            return ('unknown', 0.0, -1)

    def save_face_crop(self, face_crop: np.ndarray, track_id: int, timestamp_ns: int):
        """
        Save high-res face crop to disk.

        Args:
            face_crop: Face crop image (BGR)
            track_id: Track ID of the person
            timestamp_ns: Timestamp in nanoseconds
        """
        try:
            # Create filename with track ID and timestamp
            filename = f"track_{track_id}_{timestamp_ns}.jpg"
            filepath = os.path.join(self.face_crops_dir, filename)

            # Save image
            cv2.imwrite(filepath, face_crop)

            self.get_logger().debug(f"Saved face crop: {filename}")

        except Exception as e:
            self.get_logger().error(f"Failed to save face crop: {e}")

    def cleanup_cache(self):
        """Remove old frames from cache."""
        now_ns = self.get_clock().now().nanoseconds
        cache_duration_ns = int(self.cache_duration_sec * 1e9)

        old_frames = [ts for ts in self.frame_cache.keys() if now_ns - ts > cache_duration_ns]
        for ts in old_frames:
            del self.frame_cache[ts]

        if len(old_frames) > 0:
            self.get_logger().debug(f"Cleaned {len(old_frames)} old frames from cache")


def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
