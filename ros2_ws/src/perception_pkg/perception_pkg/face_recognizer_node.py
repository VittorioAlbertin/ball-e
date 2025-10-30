"""
Face Recognizer Node (Service-Based) - InsightFace

DESCRIPTION:
    The FaceRecognizer node is responsible for generating face embeddings using the
    InsightFace buffalo_l model. It operates as a service, responding to synchronous
    embedding generation requests from the PersonStateManager. It extracts face crops
    from the high-resolution camera stream and generates 512-dimensional embedding vectors.

RESPONSIBILITIES:
    1. Cache high-res camera frames
    2. Provide GenerateEmbedding service for on-demand embedding generation
    3. Scale face bbox from low-res (640x360) to high-res (1920x1080) coordinates
    4. Extract face crops from high-res images
    5. Generate face embeddings using InsightFace buffalo_l model
    6. Return embeddings synchronously

MODEL DETAILS:
    - Model: InsightFace buffalo_l (ArcFace-based recognition)
    - Input: Face crop (any size, handled internally by InsightFace)
    - Output: 512-dimensional embedding vector (L2 normalized)
    - Much more reliable than raw FaceNet ONNX models

SUBSCRIPTIONS:
    - /camera/image_raw (sensor_msgs/Image): High-res camera stream (1920x1080)

SERVICES:
    - /face_recognition/generate_embedding (GenerateEmbedding): Generate face embedding

PARAMETERS:
    - use_gpu (bool): Use GPU acceleration if available [default: true]
    - low_res_width (int): Low-res image width for bbox scaling [default: 640]
    - low_res_height (int): Low-res image height for bbox scaling [default: 360]
    - high_res_width (int): High-res image width for bbox scaling [default: 1920]
    - high_res_height (int): High-res image height for bbox scaling [default: 1080]
    - model_dir (str): Directory for InsightFace models [default: ~/.insightface/models]

DATA FLOW:
    1. Receive service request (track_id, face bbox in low-res, timestamp)
    2. Lookup high-res frame in cache by timestamp
    3. Scale bbox from low-res to high-res
    4. Crop face from high-res image
    5. Use InsightFace to detect and extract embedding
    6. Return embedding synchronously

COORDINATE SCALING:
    Input bbox is in low-res coordinates (640x360), must be scaled to high-res (1920x1080):
    - scale_x = 1920 / 640 = 3.0
    - scale_y = 1080 / 360 = 3.0
    - high_res_x = low_res_x * scale_x
    - high_res_w = low_res_w * scale_x

GPU ACCELERATION:
    - Attempts to use CUDAExecutionProvider if available
    - Falls back to CPUExecutionProvider if GPU not available

AUTHOR: Vittorio Albertin
DATE: 2025-10-30
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from msgs_interfaces.srv import GenerateEmbedding
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

# Import InsightFace
from insightface.app import FaceAnalysis

# Model directory in perception_pkg
MODELS_DIR = "/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models"


class FaceRecognizerNode(Node):
    def __init__(self):
        super().__init__('face_recognizer_node')

        # Declare parameters
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('low_res_width', 640)
        self.declare_parameter('low_res_height', 360)
        self.declare_parameter('high_res_width', 1920)
        self.declare_parameter('high_res_height', 1080)
        self.declare_parameter('model_dir', os.path.expanduser('~/.insightface/models'))

        # Get parameters
        use_gpu = self.get_parameter('use_gpu').value
        self.low_res_width = self.get_parameter('low_res_width').value
        self.low_res_height = self.get_parameter('low_res_height').value
        self.high_res_width = self.get_parameter('high_res_width').value
        self.high_res_height = self.get_parameter('high_res_height').value
        model_dir = self.get_parameter('model_dir').value

        # Compute scaling factors
        self.scale_x = self.high_res_width / self.low_res_width
        self.scale_y = self.high_res_height / self.low_res_height

        # Setup execution providers (GPU/CPU)
        providers = ['CPUExecutionProvider']  # Default to CPU

        if use_gpu:
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self.get_logger().info("GPU acceleration enabled (CUDA)")
                elif 'TensorrtExecutionProvider' in available_providers:
                    providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
                    self.get_logger().info("GPU acceleration enabled (TensorRT)")
                else:
                    self.get_logger().info("GPU not available, using CPU")
            except Exception:
                self.get_logger().info("CUDA not available, using CPU")
        else:
            self.get_logger().info("GPU disabled, using CPU")

        # Initialize InsightFace FaceAnalysis
        # This will download buffalo_l model if not present
        self.get_logger().info("Loading InsightFace buffalo_l model...")

        # Set model directory to use downloaded model if present
        os.environ['INSIGHTFACE_HOME'] = model_dir

        try:
            self.face_analyzer = FaceAnalysis(
                name='buffalo_l',
                root=model_dir,
                providers=providers
            )
            self.face_analyzer.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
            self.get_logger().info(f"✓ InsightFace buffalo_l model loaded successfully")
            self.get_logger().info(f"  Model directory: {model_dir}")
            self.get_logger().info(f"  Providers: {providers}")
        except Exception as e:
            self.get_logger().error(f"Failed to load InsightFace model: {e}")
            raise RuntimeError(f"Failed to load InsightFace model: {e}")

        # Bridge for ROS-CV conversion
        self.bridge = CvBridge()

        # Cache for high-res frames
        self.frame_cache = {}  # {timestamp_ns: Image}
        self.cache_duration_sec = 3.0  # Keep frames for 3 seconds (for enrollment)

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Service server
        self.generate_embedding_service = self.create_service(
            GenerateEmbedding,
            '/face_recognition/generate_embedding',
            self.generate_embedding_callback
        )

        # Cleanup timer
        self.create_timer(2, self.cleanup_cache)

        self.get_logger().info("FaceRecognizerNode initialized (InsightFace buffalo_l)")
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

    def generate_embedding_callback(self, request, response):
        """
        Service callback for embedding generation.

        Args:
            request: GenerateEmbedding.Request containing face bbox in low-res
            response: GenerateEmbedding.Response to populate

        Returns:
            GenerateEmbedding.Response with embedding or failure
        """
        # Convert timestamp to nanoseconds for cache key
        timestamp_ns = request.header.stamp.sec * 1_000_000_000 + request.header.stamp.nanosec

        # Check if frame is available in cache
        if timestamp_ns not in self.frame_cache:
            response.success = False
            response.message = f"Frame not found in cache for timestamp {timestamp_ns}"
            self.get_logger().warning(response.message)
            return response

        frame_msg = self.frame_cache[timestamp_ns]

        try:
            # Get high-res frame
            frame = self.bridge.imgmsg_to_cv2(frame_msg, desired_encoding='bgr8')

            # Scale bbox from low-res to high-res
            high_res_bbox = self.scale_bbox(
                request.face_x, request.face_y, request.face_w, request.face_h
            )

            self.get_logger().debug(
                f"Track {request.track_id} - Low-res bbox: ({request.face_x}, {request.face_y}, {request.face_w}, {request.face_h})"
            )
            self.get_logger().debug(
                f"Track {request.track_id} - High-res bbox: {high_res_bbox}, Frame shape: {frame.shape}"
            )

            # Extract face crop
            face_crop = self.extract_face(frame, high_res_bbox)

            if face_crop is None:
                response.success = False
                response.message = f"Failed to extract face for track {request.track_id}"
                self.get_logger().warning(response.message)
                return response

            # Generate embedding
            embedding = self.generate_embedding(face_crop)

            # Populate response
            response.success = True
            response.embedding = embedding.tolist()
            response.message = "Embedding generated successfully"

            self.get_logger().info(
                f"Embedding generated for track {request.track_id} " +
                f"(embedding dim: {len(embedding)})"
            )

            return response

        except Exception as e:
            response.success = False
            response.message = f"Embedding generation failed: {str(e)}"
            self.get_logger().error(response.message)
            return response

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
        Generate 512-dim face embedding using InsightFace.

        Args:
            face: Face crop (BGR)

        Returns:
            np.ndarray: 512-dim embedding vector (L2 normalized)
        """
        # Use InsightFace to detect face and extract embedding
        # InsightFace handles all preprocessing internally
        faces = self.face_analyzer.get(face)

        if len(faces) == 0:
            self.get_logger().warning(f"[EMBEDDING] No face detected in crop")
            # Return zero embedding as fallback
            return np.zeros(512, dtype=np.float32)

        # Use the first detected face
        if len(faces) > 1:
            self.get_logger().debug(f"[EMBEDDING] Multiple faces detected ({len(faces)}), using first one")

        embedding = faces[0].embedding

        # Normalize embedding (InsightFace embeddings are already normalized, but do it anyway for consistency)
        norm = np.linalg.norm(embedding)
        embedding_normalized = embedding / norm if norm > 0 else embedding

        # Log embedding stats
        self.get_logger().info(f"[EMBEDDING] Generated embedding: dim={len(embedding_normalized)}, norm={np.linalg.norm(embedding_normalized):.6f}")
        self.get_logger().info(f"[EMBEDDING] Stats: min={embedding_normalized.min():.6f}, max={embedding_normalized.max():.6f}, mean={embedding_normalized.mean():.6f}, std={embedding_normalized.std():.6f}")

        return embedding_normalized.astype(np.float32)

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
