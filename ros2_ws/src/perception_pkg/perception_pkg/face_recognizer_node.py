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
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from msgs_interfaces.srv import GenerateEmbedding
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import time
import threading

# Import InsightFace
from insightface.app import FaceAnalysis

# Model directory outside of package (to avoid colcon build issues)
MODELS_DIR = "/ball-e/ros2_ws/models"


class FaceRecognizerNode(Node):
    def __init__(self):
        super().__init__('face_recognizer_node')

        # Declare parameters
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('low_res_width', 640)
        self.declare_parameter('low_res_height', 360)
        self.declare_parameter('high_res_width', 1920)
        self.declare_parameter('high_res_height', 1080)
        self.declare_parameter('model_dir', MODELS_DIR)

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

        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # InsightFace expects: root/models/buffalo_l/
        # So if model_dir = "/ball-e/ros2_ws/models", root should be "/ball-e/ros2_ws"
        insightface_root = os.path.dirname(model_dir)

        try:
            # Use root parameter to specify where models are stored
            # InsightFace will look for models in root/models/buffalo_l/
            self.face_analyzer = FaceAnalysis(
                name='buffalo_l',
                root=insightface_root,
                providers=providers
            )
            self.face_analyzer.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
            self.get_logger().info(f"✓ InsightFace buffalo_l model loaded successfully")
            self.get_logger().info(f"  Root directory: {insightface_root}")
            self.get_logger().info(f"  Model path: {model_dir}/buffalo_l/")
            self.get_logger().info(f"  Providers: {providers}")
        except Exception as e:
            self.get_logger().error(f"Failed to load InsightFace model: {e}")
            raise RuntimeError(f"Failed to load InsightFace model: {e}")

        # Bridge for ROS-CV conversion
        self.bridge = CvBridge()

        # Cache for high-res frames
        self.frame_cache = {}  # {timestamp_ns: Image}
        self.cache_duration_sec = 5.0  # Keep frames for 5 seconds (match detector cache)
        self.max_cache_size = 100  # Max frames to keep (prevents unbounded growth)

        # Pinned frames cache (explicitly requested frames for identification)
        self.pinned_frames = {}  # {timestamp_ns: (Image, pin_time)}
        self.pinned_max_age = 30.0  # Auto-expire pinned frames after 30 seconds

        # Thread-safe lock for cache operations
        self.cache_lock = threading.Lock()

        # Create callback groups for multi-threaded execution
        # Reentrant group for subscriptions (allow concurrent frame caching)
        self.subscription_group = ReentrantCallbackGroup()
        # Mutually exclusive group for service (heavy computation, one at a time)
        self.service_group = MutuallyExclusiveCallbackGroup()

        # Subscriptions - use reentrant group so frames keep arriving during service calls
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            30,  # Increased from 10 to handle processing delays
            callback_group=self.subscription_group
        )

        # Pin/release frame subscriptions
        self.pin_sub = self.create_subscription(
            Header,
            '/frame_cache/pin_timestamp',
            self.pin_frame_callback,
            10,
            callback_group=self.subscription_group
        )
        self.release_sub = self.create_subscription(
            Header,
            '/frame_cache/release_timestamp',
            self.release_frame_callback,
            10,
            callback_group=self.subscription_group
        )

        # Service server - use separate group for heavy computation
        self.generate_embedding_service = self.create_service(
            GenerateEmbedding,
            '/face_recognition/generate_embedding',
            self.generate_embedding_callback,
            callback_group=self.service_group
        )

        # Cleanup timer - use subscription group so it runs even during service calls
        self.create_timer(0.5, self.cleanup_cache, callback_group=self.subscription_group)

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
        with self.cache_lock:
            self.frame_cache[timestamp_ns] = msg

        # Log first frame received (diagnostic)
        if not hasattr(self, '_first_frame_logged'):
            self._first_frame_logged = True
            self.get_logger().info(
                f"First high-res frame received: {msg.width}x{msg.height}, "
                f"timestamp {timestamp_ns}, cache now has {len(self.frame_cache)} frames"
            )

    def pin_frame_callback(self, msg: Header):
        """
        Pin a specific frame timestamp for identification.
        Waits for the frame to arrive if not yet in cache.

        Args:
            msg: Header with timestamp to pin
        """
        timestamp_ns = msg.stamp.sec * 1_000_000_000 + msg.stamp.nanosec

        # Check if frame is in rolling cache (with lock)
        with self.cache_lock:
            if timestamp_ns in self.frame_cache:
                self.pinned_frames[timestamp_ns] = (self.frame_cache[timestamp_ns], time.time())
                self.get_logger().info(f"[PIN] Pinned frame timestamp {timestamp_ns} (immediate)")
                return

            # Log initial cache state
            if self.frame_cache:
                cache_timestamps = sorted(self.frame_cache.keys())
                newest = cache_timestamps[-1]
                initial_delay_ms = (timestamp_ns - newest) / 1_000_000
                self.get_logger().info(
                    f"[PIN] Waiting for frame {timestamp_ns}, currently {initial_delay_ms:.1f}ms behind newest"
                )
            else:
                self.get_logger().warning(f"[PIN] Cache is EMPTY, waiting for frame {timestamp_ns}")

        # Frame not yet in cache - wait for it to arrive (reduced timeout)
        max_wait_ms = 100  # Wait up to 100ms (reduced from 500ms)
        wait_interval_ms = 5  # Check more frequently
        waited_ms = 0

        while waited_ms < max_wait_ms:
            time.sleep(wait_interval_ms / 1000.0)
            waited_ms += wait_interval_ms

            with self.cache_lock:
                if timestamp_ns in self.frame_cache:
                    self.pinned_frames[timestamp_ns] = (self.frame_cache[timestamp_ns], time.time())
                    self.get_logger().info(f"[PIN] Pinned frame timestamp {timestamp_ns} (waited {waited_ms}ms)")
                    return

        # Still not found after waiting
        with self.cache_lock:
            if self.frame_cache:
                cache_timestamps = sorted(self.frame_cache.keys())
                newest = cache_timestamps[-1]
                delay_ms = (timestamp_ns - newest) / 1_000_000
                self.get_logger().warning(
                    f"[PIN] FAILED to pin timestamp {timestamp_ns} after waiting {waited_ms}ms. "
                    f"Latest in cache: {newest} ({delay_ms:.1f}ms behind). "
                    f"Cache size: {len(self.frame_cache)} frames"
                )
            else:
                self.get_logger().warning(
                    f"[PIN] Cannot pin timestamp {timestamp_ns} - cache is EMPTY after waiting {waited_ms}ms"
                )

    def release_frame_callback(self, msg: Header):
        """
        Release a pinned frame timestamp.

        Args:
            msg: Header with timestamp to release
        """
        timestamp_ns = msg.stamp.sec * 1_000_000_000 + msg.stamp.nanosec

        with self.cache_lock:
            if timestamp_ns in self.pinned_frames:
                del self.pinned_frames[timestamp_ns]
                self.get_logger().debug(f"[RELEASE] Released pinned frame {timestamp_ns}")

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

        # Check if frame is available in pinned cache first, then rolling cache
        frame_msg = None
        with self.cache_lock:
            if timestamp_ns in self.pinned_frames:
                frame_msg, _ = self.pinned_frames[timestamp_ns]
                self.get_logger().debug(f"Track {request.track_id}: Found frame in pinned cache")
            elif timestamp_ns in self.frame_cache:
                frame_msg = self.frame_cache[timestamp_ns]
                self.get_logger().debug(f"Track {request.track_id}: Found frame in rolling cache")
            else:
                # Diagnostic: show what timestamps ARE in cache
                diag_msg = f"Frame not found for timestamp {timestamp_ns}. "
                if self.pinned_frames:
                    diag_msg += f"Pinned cache has: {list(self.pinned_frames.keys())}. "
                else:
                    diag_msg += "Pinned cache is EMPTY. "
                if self.frame_cache:
                    cache_timestamps = sorted(self.frame_cache.keys())
                    diag_msg += f"Rolling cache has {len(self.frame_cache)} frames, newest: {cache_timestamps[-1]}"
                else:
                    diag_msg += "Rolling cache is EMPTY."

                response.success = False
                response.message = diag_msg
                self.get_logger().warning(response.message)
                return response

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
        """Remove old frames from rolling cache and expired pinned frames."""
        now_ns = self.get_clock().now().nanoseconds
        cache_duration_ns = int(self.cache_duration_sec * 1e9)

        with self.cache_lock:
            # Clean rolling cache - remove old frames
            old_frames = [ts for ts in self.frame_cache.keys() if now_ns - ts > cache_duration_ns]
            for ts in old_frames:
                del self.frame_cache[ts]

            if len(old_frames) > 0:
                self.get_logger().debug(f"Cleaned {len(old_frames)} old frames from rolling cache")

            # Enforce max cache size to prevent unbounded growth
            if len(self.frame_cache) > self.max_cache_size:
                sorted_ts = sorted(self.frame_cache.keys())
                to_remove = sorted_ts[:-self.max_cache_size]  # Keep newest
                for ts in to_remove:
                    del self.frame_cache[ts]
                if to_remove:
                    self.get_logger().warning(
                        f"[CLEANUP] Cache overflow! Removed {len(to_remove)} frames, keeping {self.max_cache_size}"
                    )

            # Clean expired pinned frames (safety net)
            now = time.time()
            expired_pins = [ts for ts, (_, pin_time) in self.pinned_frames.items()
                            if now - pin_time > self.pinned_max_age]
            for ts in expired_pins:
                del self.pinned_frames[ts]
                self.get_logger().warning(f"[CLEANUP] Auto-expired pinned frame {ts} after {self.pinned_max_age}s")


def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognizerNode()

    # Use MultiThreadedExecutor to allow concurrent execution
    # This prevents frame caching from blocking during heavy computation
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
