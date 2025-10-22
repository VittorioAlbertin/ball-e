import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import ros2_numpy as rnp
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from msgs_interfaces.msg import FaceRecognition
from msgs_interfaces.srv import RecognizeFace
import onnxruntime as ort
import os
import urllib.request
from pathlib import Path
import time

class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection_node')
        
        # Parameters
        # Ultra-lightweight face detection (320x240, ~1MB)
        #'face_model_url': 'https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx'
        self.declare_parameter('yolo_person_class_id', 0)
        self.declare_parameter('face_model_path', '/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/yoloface.onnx')
        self.declare_parameter('face_model_url', 'https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx')
        self.declare_parameter('embedding_model_path', '/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/facenet.onnx')
        self.declare_parameter('embedding_model_url', 'https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx')
        self.declare_parameter('face_confidence_threshold', 0.6)  # Lowered from 0.5 to 0.3
        self.declare_parameter('recognition_threshold', 0.6)
        self.declare_parameter('nms_iou_threshold', 0.3)  # NMS threshold for removing duplicate detections
        
        self.yolo_person_class_id = self.get_parameter('yolo_person_class_id').value
        self.face_model_path = self.get_parameter('face_model_path').value
        self.face_model_url = self.get_parameter('face_model_url').value
        self.embedding_model_path = self.get_parameter('embedding_model_path').value
        self.embedding_model_url = self.get_parameter('embedding_model_url').value
        self.face_confidence_threshold = self.get_parameter('face_confidence_threshold').value
        self.recognition_threshold = self.get_parameter('recognition_threshold').value
        self.nms_iou_threshold = self.get_parameter('nms_iou_threshold').value
        
        # Download models if they don't exist
        self._ensure_model_exists(self.face_model_path, self.face_model_url, "face detection")
        self._ensure_model_exists(self.embedding_model_path, self.embedding_model_url, "face embedding")

        # Load lightweight ONNX models
        try:
            self.get_logger().info("Loading ONNX models...")
            self.face_session = ort.InferenceSession(self.face_model_path)
            self.embedding_session = ort.InferenceSession(self.embedding_model_path)
            
            # Get model input names and shapes
            self.face_input_name = self.face_session.get_inputs()[0].name
            self.face_input_shape = self.face_session.get_inputs()[0].shape
            
            self.embedding_input_name = self.embedding_session.get_inputs()[0].name
            self.embedding_input_shape = self.embedding_session.get_inputs()[0].shape
            
            self.get_logger().info(f"✓ Loaded face detection model: {self.face_model_path}")
            self.get_logger().info(f"  Input shape: {self.face_input_shape}")
            self.get_logger().info(f"✓ Loaded embedding model: {self.embedding_model_path}")
            self.get_logger().info(f"  Input shape: {self.embedding_input_shape}")
        except Exception as e:
            self.get_logger().error(f"Failed to load models: {e}")
            raise
        
        # Current image for coordinate transformation
        self.current_image = None
        self.current_image_header = None

        # Store recognition results for visualization
        self.recognition_results = {}  # {(cx, cy): (name, confidence)}
        
        # Service client for face recognition
        self.face_recognition_client = self.create_client(RecognizeFace, 'people_db/recognize_face')
        while not self.face_recognition_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for recognize_face service...')
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.detections_sub = self.create_subscription(
            Detection2DArray,
            '/yolo/detections',
            self.detections_callback,
            10
        )
        
        # Publishers
        self.face_detections_pub = self.create_publisher(
            Detection2DArray,
            'face/detections',
            10
        )
        
        self.face_recognition_pub = self.create_publisher(
            FaceRecognition,
            'face/recognition',
            10
        )
        
        self.face_image_pub = self.create_publisher(
            Image,
            'face/debug_image',
            5
        )
        
        self.get_logger().info("Face Detection Node initialized")

    def _nms(self, boxes, scores, iou_threshold):
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(boxes) == 0:
            return [], []

        boxes = np.array(boxes)
        scores = np.array(scores)

        # Get coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)

        # Sort by scores
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Calculate IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # Keep only boxes with IoU less than threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return boxes[keep].tolist(), scores[keep].tolist()

    def _ensure_model_exists(self, model_path, model_url, model_name):
        """Download model if it doesn't exist locally"""
        if os.path.exists(model_path):
            self.get_logger().info(f"✓ {model_name} model found: {model_path}")
            return
        
        self.get_logger().info(f"⚠ {model_name} model not found, downloading from {model_url}")
        
        try:
            # Create directory if it doesn't exist
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                if block_num % 50 == 0:  # Log every ~5%
                    self.get_logger().info(f"  Downloading {model_name}: {percent:.1f}%")
            
            urllib.request.urlretrieve(model_url, model_path, reporthook=report_progress)
            self.get_logger().info(f"✓ Successfully downloaded {model_name} model")
            
        except Exception as e:
            self.get_logger().error(f"Failed to download {model_name} model: {e}")
            self.get_logger().error(f"Please manually download from: {model_url}")
            self.get_logger().error(f"And place it at: {model_path}")
            raise

    def image_callback(self, msg):
        """Store current image for coordinate transformation"""
        try:
            self.current_image = rnp.numpify(msg)
            # Image is already in BGR format from camera
            self.current_image_header = msg.header
            self.get_logger().debug(f"Received image: {self.current_image.shape}")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
    
    def detections_callback(self, msg):
        """Process YOLO detections and find faces in person ROIs"""
        if self.current_image is None:
            self.get_logger().warning("No current image available")
            return

        face_detections = Detection2DArray()
        face_detections.header = msg.header
        debug_image = self.current_image.copy()

        # Don't clear old recognition results - keep them for continuity
        # Only clear results older than 5 seconds
        current_time = time.time()
        if not hasattr(self, 'recognition_timestamps'):
            self.recognition_timestamps = {}

        # Clean up old entries
        keys_to_remove = []
        for key in list(self.recognition_results.keys()):
            if key not in self.recognition_timestamps or current_time - self.recognition_timestamps[key] > 5.0:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.recognition_results.pop(key, None)
            self.recognition_timestamps.pop(key, None)

        self.get_logger().debug(f"Received {len(msg.detections)} YOLO detections")

        # Filter for person detections
        person_count = 0
        for det in msg.detections:
            if not det.results:
                continue
            
            # Check if detection is a person (class_id is a string from YOLO)
            class_id = det.results[0].hypothesis.class_id
            if class_id != str(self.yolo_person_class_id):
                self.get_logger().debug(f"Skipping detection with class_id={class_id} (not person)")
                continue

            person_count += 1
            self.get_logger().debug(f"Processing person detection {person_count}")

            # Extract ROI from bounding box
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            sx = det.bbox.size_x
            sy = det.bbox.size_y
            
            x_min = int(max(0, cx - sx / 2))
            y_min = int(max(0, cy - sy / 2))
            x_max = int(min(self.current_image.shape[1], cx + sx / 2))
            y_max = int(min(self.current_image.shape[0], cy + sy / 2))
            
            roi = self.current_image[y_min:y_max, x_min:x_max]

            self.get_logger().debug(f"ROI size: {roi.shape}, bounds: [{x_min},{y_min},{x_max},{y_max}]")

            if roi.size == 0:
                self.get_logger().warning("Empty ROI, skipping")
                continue

            # Detect faces in the ROI
            faces, face_confidences = self.detect_faces(roi)

            # Apply NMS to remove duplicate detections
            if len(faces) > 0:
                faces, face_confidences = self._nms(faces, face_confidences, self.nms_iou_threshold)
                self.get_logger().info(f"After NMS: {len(faces)} faces")

            # Process each detected face
            for face_bbox, face_conf in zip(faces, face_confidences):
                if face_conf < self.face_confidence_threshold:
                    continue
                
                # Convert face coordinates from ROI to original frame
                fx_min, fy_min, fx_max, fy_max = face_bbox
                face_x_min = x_min + fx_min
                face_y_min = y_min + fy_min
                face_x_max = x_min + fx_max
                face_y_max = y_min + fy_max
                
                # Extract face patch for embedding
                face_patch = self.current_image[face_y_min:face_y_max, face_x_min:face_x_max]
                
                if face_patch.size == 0:
                    continue
                
                # Get face embedding
                embedding = self.get_face_embedding(face_patch)
                
                # Create Detection2D message in original frame coordinates
                det_face = Detection2D()
                det_face.header = msg.header
                det_face.bbox.center.position.x = float((face_x_min + face_x_max) / 2.0)
                det_face.bbox.center.position.y = float((face_y_min + face_y_max) / 2.0)
                det_face.bbox.size_x = float(face_x_max - face_x_min)
                det_face.bbox.size_y = float(face_y_max - face_y_min)
                
                # Store face confidence
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(1)  # Face class ID
                hyp.hypothesis.score = float(face_conf)
                det_face.results.append(hyp)
                
                face_detections.detections.append(det_face)

                # Store as detecting initially (for fast visualization)
                bbox_key = (face_x_min, face_y_min, face_x_max, face_y_max)

                # Check if we already have a result for a similar bbox (within 20 pixels)
                existing_key = None
                for key in self.recognition_results.keys():
                    if abs(key[0] - face_x_min) < 20 and abs(key[1] - face_y_min) < 20:
                        existing_key = key
                        break

                if existing_key is None:
                    # New face detection
                    self.recognition_results[bbox_key] = ("Detecting...", face_conf)
                    self.recognition_timestamps[bbox_key] = current_time
                else:
                    # Update existing detection position
                    bbox_key = existing_key
                    self.recognition_timestamps[bbox_key] = current_time

                # Call recognition service asynchronously (non-blocking)
                self.recognize_and_publish_face_async(
                    msg.header,
                    embedding,
                    face_conf,
                    det_face.bbox.center.position.x,
                    det_face.bbox.center.position.y,
                    det_face.bbox.size_x,
                    det_face.bbox.size_y,
                    face_x_min,
                    face_y_min,
                    face_x_max,
                    face_y_max
                )
        
        # Draw all face detections with recognition results
        for (x_min, y_min, x_max, y_max), (name, conf) in self.recognition_results.items():
            # Choose color based on recognition status
            if name == "Detecting...":
                color = (255, 255, 0)  # Yellow/cyan for detecting
            elif name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for recognized

            # Draw bounding box
            cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), color, 2)

            # Draw label with background
            label = f"{name}: {conf:.2f}" if name != "Unknown" else f"Unknown: {conf:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            label_pt1 = (x_min, y_min - label_height - baseline - 5)
            label_pt2 = (x_min + label_width, y_min)
            cv2.rectangle(debug_image, label_pt1, label_pt2, color, -1)
            cv2.putText(
                debug_image, label,
                (x_min, y_min - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        # Publish face detections
        if len(face_detections.detections) > 0:
            self.face_detections_pub.publish(face_detections)

        # Always publish debug image
        try:
            debug_msg = rnp.msgify(Image, debug_image, encoding='bgr8')
            debug_msg.header = msg.header
            self.face_image_pub.publish(debug_msg)
            self.get_logger().debug("Published debug image")
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def recognize_and_publish_face_async(self, header, embedding, confidence, center_x, center_y, size_x, size_y, x_min, y_min, x_max, y_max):
        """Call recognition service asynchronously (non-blocking) and publish results"""
        try:
            # Create service request
            request = RecognizeFace.Request()
            request.face_embedding = embedding.tolist()
            request.threshold = self.recognition_threshold

            # Call service asynchronously (don't wait)
            future = self.face_recognition_client.call_async(request)

            # Add callback to handle response when it arrives
            def handle_response(future):
                try:
                    response = future.result()
                except Exception as e:
                    self.get_logger().error(f"Recognition service call failed: {e}")
                    bbox_key = (x_min, y_min, x_max, y_max)

                    # Find existing key if bbox coordinates changed slightly
                    for key in list(self.recognition_results.keys()):
                        if abs(key[0] - x_min) < 20 and abs(key[1] - y_min) < 20:
                            bbox_key = key
                            break

                    self.recognition_results[bbox_key] = ("Unknown", confidence)
                    self.recognition_timestamps[bbox_key] = time.time()
                    return

                if response is not None:
                    # Create recognition message
                    recognition_msg = FaceRecognition()
                    recognition_msg.header = header
                    recognition_msg.face_embedding = embedding.tolist()
                    recognition_msg.bbox_center_x = center_x
                    recognition_msg.bbox_center_y = center_y
                    recognition_msg.bbox_size_x = size_x
                    recognition_msg.bbox_size_y = size_y
                    recognition_msg.confidence = confidence

                    # Fill in recognition results
                    recognition_msg.found = response.found
                    recognition_msg.person_id = response.person_id
                    recognition_msg.name = response.name
                    recognition_msg.last_seen = response.last_seen
                    recognition_msg.interaction_count = response.interaction_count
                    recognition_msg.preferences_json = response.preferences_json
                    recognition_msg.notes = response.notes
                    recognition_msg.message = response.message

                    # Publish recognition result
                    self.face_recognition_pub.publish(recognition_msg)

                    # Store result for visualization
                    name = response.name if response.found else "Unknown"
                    bbox_key = (x_min, y_min, x_max, y_max)

                    # Find existing key if bbox coordinates changed slightly
                    for key in list(self.recognition_results.keys()):
                        if abs(key[0] - x_min) < 20 and abs(key[1] - y_min) < 20:
                            bbox_key = key
                            break

                    self.recognition_results[bbox_key] = (name, confidence)
                    self.recognition_timestamps[bbox_key] = time.time()

                    if response.found:
                        self.get_logger().info(f"Face recognized: {response.name} (ID: {response.person_id})")
                    else:
                        self.get_logger().info("Face detected but not recognized (unknown person)")
                else:
                    self.get_logger().warning("Recognition service call failed")
                    bbox_key = (x_min, y_min, x_max, y_max)

                    # Find existing key if bbox coordinates changed slightly
                    for key in list(self.recognition_results.keys()):
                        if abs(key[0] - x_min) < 20 and abs(key[1] - y_min) < 20:
                            bbox_key = key
                            break

                    self.recognition_results[bbox_key] = ("Unknown", confidence)
                    self.recognition_timestamps[bbox_key] = time.time()

            # Attach callback to future
            future.add_done_callback(handle_response)

        except Exception as e:
            self.get_logger().error(f"Error during recognition: {e}")
            # Store as unknown for visualization
            self.recognition_results[(x_min, y_min, x_max, y_max)] = ("Unknown", confidence)
    
    def detect_faces(self, image):
        """
        Detect faces in image using lightweight ONNX model.
        Returns list of face bboxes and confidences.
        """
        try:
            # Prepare input
            h, w = self.face_input_shape[2], self.face_input_shape[3]
            resized = cv2.resize(image, (w, h))

            # Normalize (adjust based on model requirements)
            input_data = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32) / 255.0

            # Run inference
            outputs = self.face_session.run(None, {self.face_input_name: input_data})

            # Debug: Log output shapes (only once per session)
            if not hasattr(self, '_logged_output_shapes'):
                self.get_logger().info(f"Face detection outputs: {len(outputs)} tensors")
                for i, output in enumerate(outputs):
                    self.get_logger().info(f"  Output {i} shape: {output.shape}")
                self._logged_output_shapes = True

            # Parse outputs (adjust based on model output format)
            faces = []
            confidences = []

            # Try to parse based on common YOLO/UltraFace formats
            # Format varies by model, need to inspect outputs
            if len(outputs) >= 2:
                # UltraFace format: scores and boxes as separate outputs
                # Check which output is which based on the last dimension
                if outputs[0].shape[2] == 2:
                    # Output 0 is scores (background, face)
                    scores = outputs[0]  # Shape: [batch, num_boxes, 2]
                    boxes = outputs[1]   # Shape: [batch, num_boxes, 4]
                else:
                    # Output 0 is boxes
                    boxes = outputs[0]   # Shape: [batch, num_boxes, 4]
                    scores = outputs[1]  # Shape: [batch, num_boxes, 2]

                # Process detections
                num_detections = min(boxes.shape[1], scores.shape[1])

                # Track max confidence for debugging
                max_conf = 0.0

                for i in range(num_detections):
                    # Get confidence (handle different score formats)
                    if len(scores.shape) == 3 and scores.shape[2] == 2:
                        conf = float(scores[0, i, 1])  # Face class probability (index 1)
                    elif len(scores.shape) == 3:
                        conf = float(scores[0, i, 0])
                    else:
                        conf = float(scores[0, i])

                    # Track max confidence
                    max_conf = max(max_conf, conf)

                    if conf > self.face_confidence_threshold:
                        # Boxes are usually in [x_min, y_min, x_max, y_max] format normalized
                        box = boxes[0, i]
                        x_min = max(0, int(box[0] * image.shape[1]))
                        y_min = max(0, int(box[1] * image.shape[0]))
                        x_max = min(image.shape[1], int(box[2] * image.shape[1]))
                        y_max = min(image.shape[0], int(box[3] * image.shape[0]))

                        # Validate box dimensions
                        if x_max > x_min and y_max > y_min:
                            faces.append([x_min, y_min, x_max, y_max])
                            confidences.append(conf)

                            self.get_logger().debug(f"Detected face: conf={conf:.3f}, box=[{x_min},{y_min},{x_max},{y_max}]")

                # Log max confidence found
                if len(faces) == 0:
                    self.get_logger().debug(f"No faces detected. Max confidence: {max_conf:.3f}, threshold: {self.face_confidence_threshold:.3f}")
                else:
                    self.get_logger().info(f"Found {len(faces)} faces, max confidence: {max_conf:.3f}")

            else:
                # Single output format (YOLO-style)
                detections = outputs[0]
                self.get_logger().info(f"Single output format, shape: {detections.shape}")

                if len(detections.shape) == 3:
                    detections = detections[0]  # Remove batch dimension

                for det in detections:
                    if len(det) >= 5:
                        conf = float(det[4])
                        if conf > self.face_confidence_threshold:
                            # Assuming format: [x_center, y_center, width, height, confidence]
                            x_c, y_c, bw, bh = det[:4]

                            # Convert to pixel coordinates in original ROI
                            x_min = max(0, int((x_c - bw / 2) * image.shape[1]))
                            y_min = max(0, int((y_c - bh / 2) * image.shape[0]))
                            x_max = min(image.shape[1], int((x_c + bw / 2) * image.shape[1]))
                            y_max = min(image.shape[0], int((y_c + bh / 2) * image.shape[0]))

                            faces.append([x_min, y_min, x_max, y_max])
                            confidences.append(conf)

            return faces, confidences

        except Exception as e:
            self.get_logger().error(f"Face detection failed: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return [], []
    
    def get_face_embedding(self, face_image):
        """
        Extract face embedding using lightweight ONNX model.
        Returns embedding vector.
        """
        try:
            h, w = self.embedding_input_shape[2], self.embedding_input_shape[3]
            resized = cv2.resize(face_image, (w, h))
            
            # Normalize
            input_data = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32) / 255.0
            
            # Run inference
            outputs = self.embedding_session.run(None, {self.embedding_input_name: input_data})
            
            # Extract embedding (usually first output)
            embedding = outputs[0][0].flatten()
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        
        except Exception as e:
            self.get_logger().error(f"Embedding extraction failed: {e}")
            return np.zeros(128)  # Return zero vector on failure

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()