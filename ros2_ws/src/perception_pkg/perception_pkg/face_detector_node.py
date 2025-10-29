"""
Face Detector Node

DESCRIPTION:
    The FaceDetector node is responsible for detecting faces within the Region of Interest
    (ROI) of tracked persons. It uses the YOLOv5 face detection model (yoloface.onnx) to
    locate faces in low-resolution camera images. This node operates on-demand, only
    processing frames when an identification request is received from the PersonStateManager.

RESPONSIBILITIES:
    1. Subscribe to identification requests from PersonStateManager
    2. Wait for low-res camera frames with matching timestamps
    3. Extract ROI based on person track bounding box
    4. Run YOLO face detection on the ROI
    5. Transform face bbox from ROI coordinates to full image coordinates
    6. Publish face detection results for face recognition processing

MODEL DETAILS:
    - Model: YOLOv5 Face Detection (yoloface.onnx)
    - Input: 640x640 RGB image, normalized [0, 1]
    - Output: [N, 6] - (x_center, y_center, width, height, confidence, class)
    - Post-processing: Non-Maximum Suppression (NMS) with IoU threshold 0.45

SUBSCRIPTIONS:
    - /camera/image_low_res (sensor_msgs/Image): Low-res camera stream (640x360)
    - /person_state/identification_request (RequestIdentification): Identification requests

PUBLICATIONS:
    - /face_detector/detections (FaceRecognition): Face bounding boxes with track IDs

PARAMETERS:
    - model_path (str): Path to yoloface.onnx model [default: models/yoloface.onnx]
    - confidence_threshold (float): Minimum face detection confidence [default: 0.5]
    - nms_threshold (float): IoU threshold for NMS [default: 0.45]
    - input_size (int): Model input size (square) [default: 640]

DATA FLOW:
    1. Receive identification request (track_id, bbox, timestamp)
    2. Wait for low-res frame with matching timestamp
    3. Crop ROI from frame using track bbox
    4. Preprocess: Resize to 640x640, normalize [0, 1]
    5. Run YOLO inference
    6. Apply NMS to detections
    7. Transform bbox from ROI to full image coordinates
    8. Publish face detection with track_id and timestamp

COORDINATE SYSTEMS:
    - Input bbox: Low-res image coordinates (640x360)
    - ROI extraction: Crop from low-res image
    - YOLO output: Relative to 640x640 model input
    - Output bbox: Low-res image coordinates (640x360)

AUTHOR: Vittorio Albertin
DATE: 2025-10-29
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from msgs_interfaces.msg import FaceRecognition, IdentificationRequest, IdentityUpdate
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
import os
import warnings

# Suppress ONNX Runtime warnings
warnings.filterwarnings('ignore', category=UserWarning, module='onnxruntime')
ort.set_default_logger_severity(3)  # 3 = ERROR level (suppress warnings)

YOLOFACE_MODEL_PATH = "/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/yoloface.onnx"


class FaceDetectorNode(Node):
    def __init__(self):
        super().__init__('face_detector_node')

        # Declare parameters
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.45)

        # Get parameters
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.nms_threshold = self.get_parameter('nms_threshold').value

        # Check if model exists
        if not os.path.exists(YOLOFACE_MODEL_PATH):
            self.get_logger().error(f"Model file not found: {YOLOFACE_MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {YOLOFACE_MODEL_PATH}")

        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(
            YOLOFACE_MODEL_PATH,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        # Get input shape from model
        input_shape = self.session.get_inputs()[0].shape
        # input_shape is typically [batch, channels, height, width]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        # Bridge for ROS-CV conversion
        self.bridge = CvBridge()

        # Cache for pending identification requests
        self.pending_requests = {}  # {timestamp_ns: request_data}

        # Cache for recent frames
        self.frame_cache = {}  # {timestamp_ns: Image}
        self.cache_duration_sec = 1.0  # Keep frames for 1 second

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_low_res',
            self.image_callback,
            10
        )

        self.request_sub = self.create_subscription(
            IdentificationRequest,
            '/person_state/identification_request',
            self.identification_request_callback,
            10
        )

        # Publications
        self.detection_pub = self.create_publisher(FaceRecognition, '/face_detector/detections', 10)

        # Cleanup timer
        self.create_timer(0.5, self.cleanup_caches)

        self.get_logger().info("FaceDetectorNode initialized")
        self.get_logger().info(f"  Model: {YOLOFACE_MODEL_PATH}")
        self.get_logger().info(f"  Confidence threshold: {self.conf_threshold}")
        self.get_logger().info(f"  NMS threshold: {self.nms_threshold}")
        self.get_logger().info(f"  Input size: {self.input_width}x{self.input_height}")

    def identification_request_callback(self, msg):
        """
        Process identification request from PersonStateManager.

        Args:
            msg: RequestIdentification.Request containing track info and timestamp
        """
        # Convert timestamp to nanoseconds for cache key
        timestamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        # Store request
        self.pending_requests[timestamp_ns] = {
            'track_id': msg.track_id,
            'bbox': (msg.bbox_x, msg.bbox_y, msg.bbox_w, msg.bbox_h),
            'header': msg.header
        }

        self.get_logger().debug(f"Identification request received for track {msg.track_id}")

        # Try to process immediately if frame already available
        if timestamp_ns in self.frame_cache:
            self.process_face_detection(timestamp_ns)

    def image_callback(self, msg: Image):
        """
        Process incoming low-res camera frames and cache them.

        Args:
            msg: Image message from low-res camera stream
        """
        # Convert timestamp to nanoseconds
        timestamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        # Cache frame
        self.frame_cache[timestamp_ns] = msg

        # Process if there's a pending request for this timestamp
        if timestamp_ns in self.pending_requests:
            self.process_face_detection(timestamp_ns)

    def process_face_detection(self, timestamp_ns: int):
        """
        Detect face in the ROI of a tracked person.

        Args:
            timestamp_ns: Timestamp key for frame and request
        """
        if timestamp_ns not in self.pending_requests or timestamp_ns not in self.frame_cache:
            return

        request = self.pending_requests[timestamp_ns]
        frame_msg = self.frame_cache[timestamp_ns]

        try:
            # Convert ROS Image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(frame_msg, desired_encoding='bgr8')

            # Extract ROI
            x, y, w, h = request['bbox']
            # Ensure ROI is within frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            if w <= 0 or h <= 0:
                self.get_logger().warning(f"Invalid ROI for track {request['track_id']}")
                del self.pending_requests[timestamp_ns]
                return

            roi = frame[y:y+h, x:x+w]

            # Detect face in ROI
            faces = self.detect_faces(roi)

            if len(faces) == 0:
                self.get_logger().debug(f"No face detected for track {request['track_id']}")
                # Publish "unknown" result to stop pending state
                self.publish_no_face_result(request)
                del self.pending_requests[timestamp_ns]
                return

            # Take the most confident face
            best_face = max(faces, key=lambda f: f[4])
            face_x, face_y, face_w, face_h, confidence = best_face

            # Transform face bbox from ROI to full image coordinates
            full_face_x = x + face_x
            full_face_y = y + face_y

            # Publish face detection
            detection_msg = FaceRecognition()
            detection_msg.header = request['header']
            detection_msg.track_id = request['track_id']
            detection_msg.face_x = int(full_face_x)
            detection_msg.face_y = int(full_face_y)
            detection_msg.face_w = int(face_w)
            detection_msg.face_h = int(face_h)
            detection_msg.detection_confidence = float(confidence)

            self.detection_pub.publish(detection_msg)

            self.get_logger().info(
                f"Face detected for track {request['track_id']} " +
                f"(confidence: {confidence:.2f})"
            )

        except Exception as e:
            self.get_logger().error(f"Face detection failed: {e}")
            # Publish "unknown" result to stop pending state
            self.publish_no_face_result(request)

        finally:
            # Remove processed request (if still exists)
            if timestamp_ns in self.pending_requests:
                del self.pending_requests[timestamp_ns]

    def detect_faces(self, image: np.ndarray) -> list:
        """
        Run YOLO face detection on image.

        Args:
            image: OpenCV image (BGR)

        Returns:
            list: List of detections [(x, y, w, h, confidence), ...]
        """
        # Preprocess
        input_tensor = self.preprocess(image)

        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Debug: log output shape on first run
        if not hasattr(self, '_logged_output_shape'):
            self.get_logger().info(f"Number of outputs: {len(outputs)}")
            for i, out in enumerate(outputs):
                self.get_logger().info(f"  Output {i} shape: {out.shape}")
            self._logged_output_shape = True

        # Handle two-output format: [scores, boxes]
        if len(outputs) == 2:
            scores = outputs[0]  # (1, N, 2) - [background_prob, face_prob]
            boxes = outputs[1]   # (1, N, 4) - [x, y, w, h]
            faces = self.postprocess_two_outputs(scores, boxes, image.shape[:2])
        else:
            # Standard YOLO format: single output with [x, y, w, h, conf, class]
            detections = outputs[0]
            faces = self.postprocess(detections, image.shape[:2])

        return faces

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO model.

        Args:
            image: OpenCV image (BGR, HxWx3)

        Returns:
            np.ndarray: Preprocessed tensor (1x3xHxW)
        """
        # Resize to model input size (width, height)
        resized = cv2.resize(image, (self.input_width, self.input_height))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Transpose to CHW format
        chw = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batch = np.expand_dims(chw, axis=0)

        return batch

    def postprocess_two_outputs(self, scores: np.ndarray, boxes: np.ndarray, original_shape: tuple) -> list:
        """
        Post-process YOLO detections with two-output format (scores and boxes separate).

        Args:
            scores: Class scores array (1, N, 2) - [background_prob, face_prob]
            boxes: Bounding boxes array (1, N, 4) - [x1, y1, x2, y2] in normalized coords
            original_shape: Original image shape (H, W)

        Returns:
            list: Filtered detections [(x, y, w, h, confidence), ...]
        """
        # Remove batch dimension
        if len(scores.shape) == 3:
            scores = scores[0]  # (N, 2)
        if len(boxes.shape) == 3:
            boxes = boxes[0]    # (N, 4)

        # Get face confidence (second column)
        face_scores = scores[:, 1]

        # Filter by confidence
        mask = face_scores > self.conf_threshold
        filtered_scores = face_scores[mask]
        filtered_boxes = boxes[mask]

        if len(filtered_boxes) == 0:
            return []

        # Log first filtered detection for debugging
        if not hasattr(self, '_logged_first_detection'):
            self.get_logger().info(f"Total detections: {len(boxes)}, After filtering: {len(filtered_boxes)}")
            self.get_logger().info(f"First filtered score: {filtered_scores[0]:.4f}")
            self.get_logger().info(f"First filtered box (raw): {filtered_boxes[0]}")
            self.get_logger().info(f"Original shape: {original_shape}, Input: {self.input_width}x{self.input_height}")
            self._logged_first_detection = True

        # Boxes are in format [x1, y1, x2, y2] NORMALIZED (0-1 range)
        # Need to denormalize by model input size, then scale to original image size

        # Denormalize: multiply by model input dimensions to get pixel coordinates
        denorm_boxes = filtered_boxes.copy()
        denorm_boxes[:, 0] *= self.input_width   # x1
        denorm_boxes[:, 1] *= self.input_height  # y1
        denorm_boxes[:, 2] *= self.input_width   # x2
        denorm_boxes[:, 3] *= self.input_height  # y2

        # Scale from model input size to original ROI size
        scale_x = original_shape[1] / self.input_width
        scale_y = original_shape[0] / self.input_height

        scaled_boxes = denorm_boxes.copy()
        scaled_boxes[:, 0] *= scale_x  # x1
        scaled_boxes[:, 1] *= scale_y  # y1
        scaled_boxes[:, 2] *= scale_x  # x2
        scaled_boxes[:, 3] *= scale_y  # y2

        # Apply NMS using xyxy format
        indices = self.non_max_suppression(scaled_boxes, filtered_scores, self.nms_threshold)

        results = []
        for idx in indices:
            box = scaled_boxes[idx]
            score = filtered_scores[idx]

            # Convert from [x1, y1, x2, y2] to [x, y, w, h]
            x1, y1, x2, y2 = box
            x = x1
            y = y1
            width = x2 - x1
            height = y2 - y1

            # Log first result for debugging
            if len(results) == 0 and not hasattr(self, '_logged_first_result'):
                self.get_logger().info(f"First result - box corners: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
                self.get_logger().info(f"First result - output: x={x:.1f}, y={y:.1f}, w={width:.1f}, h={height:.1f}, score={score:.4f}")
                self._logged_first_result = True

            results.append((x, y, width, height, score))

        return results

    def postprocess(self, detections: np.ndarray, original_shape: tuple) -> list:
        """
        Post-process YOLO detections with NMS.

        Args:
            detections: YOLO output array
            original_shape: Original image shape (H, W)

        Returns:
            list: Filtered detections [(x, y, w, h, confidence), ...]
        """
        # Handle different output shapes
        if len(detections.shape) == 3:
            detections = detections[0]  # Remove batch dimension

        # Check if output has expected format
        if detections.shape[1] < 5:
            self.get_logger().error(f"Unexpected YOLO output shape: {detections.shape}. Expected at least 5 columns.")
            return []

        # Filter by confidence
        scores = detections[:, 4]
        mask = scores > self.conf_threshold
        filtered = detections[mask]

        if len(filtered) == 0:
            return []

        # Extract bounding boxes and scores
        boxes = filtered[:, :4]  # [x_center, y_center, width, height]
        scores = filtered[:, 4]

        # Convert from center format to corner format for NMS
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # Apply NMS
        indices = self.non_max_suppression(boxes_xyxy, scores, self.nms_threshold)

        # Scale boxes to original image size
        scale_x = original_shape[1] / self.input_width
        scale_y = original_shape[0] / self.input_height

        results = []
        for idx in indices:
            box = boxes[idx]
            score = scores[idx]

            # Convert to (x, y, w, h) and scale
            x_center = box[0] * scale_x
            y_center = box[1] * scale_y
            width = box[2] * scale_x
            height = box[3] * scale_y

            x = x_center - width / 2
            y = y_center - height / 2

            results.append((x, y, width, height, score))

        return results

    def non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list:
        """
        Apply Non-Maximum Suppression.

        Args:
            boxes: Array of boxes in xyxy format [N, 4]
            scores: Array of scores [N]
            threshold: IoU threshold

        Returns:
            list: Indices of boxes to keep
        """
        # Sort by score descending
        order = np.argsort(scores)[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU with remaining boxes
            ious = self.compute_iou(boxes[i], boxes[order[1:]])

            # Remove boxes with IoU > threshold
            mask = ious <= threshold
            order = order[1:][mask]

        return keep

    def compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Compute IoU between one box and multiple boxes.

        Args:
            box: Single box [4]
            boxes: Multiple boxes [N, 4]

        Returns:
            np.ndarray: IoU values [N]
        """
        # Intersection area
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection

        return intersection / (union + 1e-6)

    def publish_no_face_result(self, request):
        """
        Publish a result indicating no face was detected.
        This prevents the track from staying in pending state forever.

        Args:
            request: Identification request data
        """
        result = IdentityUpdate()
        result.header = request['header']
        result.track_id = request['track_id']
        result.identity = 'unknown'
        result.confidence = 0.0
        result.person_id = -1

        # Publish to face recognizer's output topic to trigger state update
        # Create publisher on first use
        if not hasattr(self, 'no_face_pub'):
            self.no_face_pub = self.create_publisher(IdentityUpdate, '/face_recognizer/results', 10)

        self.no_face_pub.publish(result)
        self.get_logger().info(f"No face detected for track {request['track_id']}, marked as unknown")

    def cleanup_caches(self):
        """Remove old frames and requests from cache."""
        now_ns = self.get_clock().now().nanoseconds
        cache_duration_ns = int(self.cache_duration_sec * 1e9)

        # Clean frame cache
        old_frames = [ts for ts in self.frame_cache.keys() if now_ns - ts > cache_duration_ns]
        for ts in old_frames:
            del self.frame_cache[ts]

        # Clean pending requests (older than 2 seconds)
        old_requests = [ts for ts in self.pending_requests.keys() if now_ns - ts > 2e9]
        for ts in old_requests:
            self.get_logger().warning(f"Timeout: identification request not processed (track {self.pending_requests[ts]['track_id']})")
            # Publish unknown result before cleaning up
            self.publish_no_face_result(self.pending_requests[ts])
            del self.pending_requests[ts]


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
