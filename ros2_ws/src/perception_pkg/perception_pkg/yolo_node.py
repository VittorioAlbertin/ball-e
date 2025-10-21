import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
import cv2
import torch
import threading
import time
import ros2_numpy as rnp
import numpy as np
import os
import warnings

# Suppress torch autocast deprecation warning from YOLOv5
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.cuda.amp.autocast.*')

MODEL_PATH = "/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/yolov5n.pt"

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")
        self.latest_msg = None
        # Subscribers and Publishers
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 1
        )
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/yolo/detections', 1
        )
        self.image_pub = self.create_publisher(
            Image, '/yolo/image_detections', 1
        )

        # Load YOLO model (offline if available)
        if os.path.exists(MODEL_PATH):
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
            self.get_logger().info(f"Loaded YOLO model from {MODEL_PATH}")
        else:
            self.get_logger().info("Downloading YOLOv5n pretrained model...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            #self.model.model.save(MODEL_PATH)
            self.get_logger().info(f"Downloaded and saved YOLO model to {MODEL_PATH}")
        self.model.to(self.device)
        self.model.eval()

        # Store COCO class names for visualization
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}

        # Async processing
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.process_frame)
        self.thread.daemon = True
        self.thread.start()

    def image_callback(self, msg):
        with self.lock:
            self.latest_msg=msg

    def process_frame(self):
        while rclpy.ok():
            with self.lock:
                #msg = self.queue.pop(0)
                msg = self.latest_msg
                self.latest_msg = None # clear buffer
            if msg is None:
                time.sleep(0.01)
                continue
            try:
                cv_image = rnp.numpify(msg)
                # Convert to torch tensor
                if cv_image.ndim == 2:  # grayscale
                    img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                results = self.model(img_rgb, size=640)#.to(self.device)  # you can adjust size
                self.publish_results(results, msg.header, cv_image)
            except Exception as e:
                self.get_logger().error(f"YOLO processing error: {e}")

    def publish_results(self, results, header, cv_image):
        # Publish Detection2DArray (bounding boxes, classes, confidence)
        det_array = Detection2DArray()
        det_array.header = header

        # Create a copy of the image for visualization
        vis_image = cv_image.copy()

        # YOLO results as a pandas dataframe: xmin, ymin, xmax, ymax, confidence, class, name
        df = results.pandas().xyxy[0]

        for idx, row in df.iterrows():
            det = Detection2D()

            # Bounding box center + size
            det.bbox.center.position.x = float((row.xmin + row.xmax) / 2.0)
            det.bbox.center.position.y = float((row.ymin + row.ymax) / 2.0)
            det.bbox.center.theta = 0.0  # No rotation for axis-aligned boxes
            det.bbox.size_x = float(row.xmax - row.xmin)
            det.bbox.size_y = float(row.ymax - row.ymin)

            # Class + confidence
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(int(row['class']))  # class_id is a string
            hyp.hypothesis.score = float(row['confidence'])
            det.results.append(hyp)

            det_array.detections.append(det)

            # Draw bounding box and label on visualization image
            class_id = int(row['class'])
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            confidence = float(row['confidence'])

            # Get color for this class
            color = self.get_color_for_class(class_id)

            # Draw bounding box
            pt1 = (int(row.xmin), int(row.ymin))
            pt2 = (int(row.xmax), int(row.ymax))
            cv2.rectangle(vis_image, pt1, pt2, color, 2)

            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            label_pt1 = (int(row.xmin), int(row.ymin) - label_height - baseline)
            label_pt2 = (int(row.xmin) + label_width, int(row.ymin))
            cv2.rectangle(vis_image, label_pt1, label_pt2, color, -1)
            cv2.putText(
                vis_image, label,
                (int(row.xmin), int(row.ymin) - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        # Publish detections
        self.detection_pub.publish(det_array)

        # Publish visualization image
        vis_msg = rnp.msgify(Image, vis_image, encoding='bgr8')
        vis_msg.header = header
        self.image_pub.publish(vis_msg)

    def get_color_for_class(self, class_id):
        """Generate a consistent BGR color for each class ID"""
        # Simple color mapping based on class_id (BGR format for OpenCV)
        colors = [
            (0, 0, 255),      # Red
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue
            (0, 255, 255),    # Yellow
            (255, 0, 255),    # Magenta
            (255, 255, 0),    # Cyan
            (0, 127, 255),    # Orange
            (255, 0, 127),    # Purple
        ]
        return colors[class_id % len(colors)]


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
