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
        self.declare_parameter('face_confidence_threshold', 0.5)
        self.declare_parameter('recognition_threshold', 0.6)
        
        self.yolo_person_class_id = self.get_parameter('yolo_person_class_id').value
        self.face_model_path = self.get_parameter('face_model_path').value
        self.face_model_url = self.get_parameter('face_model_url').value
        self.embedding_model_path = self.get_parameter('embedding_model_path').value
        self.embedding_model_url = self.get_parameter('embedding_model_url').value
        self.face_confidence_threshold = self.get_parameter('face_confidence_threshold').value
        self.recognition_threshold = self.get_parameter('recognition_threshold').value
        
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
        
        # Service client for face recognition
        self.face_recognition_client = self.create_client(RecognizeFace, 'people_db/recognize_face')
        while not self.face_recognition_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for recognize_face service...')
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        
        self.detections_sub = self.create_subscription(
            Detection2DArray,
            'yolo/detections',
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
            # Convert RGB to BGR if needed
            if len(self.current_image.shape) == 3 and self.current_image.shape[2] == 3:
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            self.current_image_header = msg.header
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
    
    def detections_callback(self, msg):
        """Process YOLO detections and find faces in person ROIs"""
        if self.current_image is None:
            return
        
        face_detections = Detection2DArray()
        face_detections.header = msg.header
        debug_image = self.current_image.copy()
        
        # Filter for person detections
        for det in msg.detections:
            if not det.results:
                continue
            
            # Check if detection is a person
            if det.results[0].hypothesis.class_id != self.yolo_person_class_id:
                continue
            
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
            
            if roi.size == 0:
                continue
            
            # Detect faces in the ROI
            faces, face_confidences = self.detect_faces(roi)
            
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
                hyp.hypotesis.score = float(face_conf)
                det_face.results.append(hyp)
                
                face_detections.detections.append(det_face)
                
                # Call recognition service
                self.recognize_and_publish_face(
                    msg.header,
                    embedding,
                    face_conf,
                    det_face.bbox.center.position.x,
                    det_face.bbox.center.position.y,
                    det_face.bbox.size_x,
                    det_face.bbox.size_y
                )
                
                # Draw on debug image (will be updated with name if recognized)
                cv2.rectangle(debug_image, (face_x_min, face_y_min), (face_x_max, face_y_max), (0, 255, 0), 2)
                cv2.putText(debug_image, f"Face: {face_conf:.2f}", (face_x_min, face_y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Publish face detections
        if len(face_detections.detections) > 0:
            self.face_detections_pub.publish(face_detections)
        
        # Publish debug image
        try:
            debug_msg = rnp.msgify(Image, debug_image, encoding='bgr8')
            debug_msg.header = msg.header
            self.face_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")
    
    def recognize_and_publish_face(self, header, embedding, confidence, center_x, center_y, size_x, size_y):
        """Call recognition service and publish results"""
        try:
            # Create service request
            request = RecognizeFace.Request()
            request.face_embedding = embedding.tolist()
            request.threshold = self.recognition_threshold
            
            # Call service synchronously
            future = self.face_recognition_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            
            if future.result() is not None:
                response = future.result()
                
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
                
                if response.found:
                    self.get_logger().info(f"Face recognized: {response.name} (ID: {response.person_id})")
                else:
                    self.get_logger().info("Face detected but not recognized (unknown person)")
            else:
                self.get_logger().warning("Recognition service call failed")
                
        except Exception as e:
            self.get_logger().error(f"Error during recognition: {e}")
    
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
            
            # Parse outputs (adjust based on model output format)
            # Assuming output format: [batch, detections, 5+num_classes] or similar
            faces = []
            confidences = []
            
            # Extract detections with NMS
            detections = outputs[0][0]  # [num_detections, 5+]
            
            for det in detections:
                conf = float(det[4])
                if conf > self.face_confidence_threshold:
                    # Assuming format: [x_center, y_center, width, height, confidence, ...]
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