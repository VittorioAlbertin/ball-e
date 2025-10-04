import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
import cv2
import torch
import threading
import time
import numpy as np
import os

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..', 'models', 'yolo_model.pt'
)


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")

        # Subscribers and Publishers
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 1
        )
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/yolo/detections', 1
        )

        # Async processing
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.process_frame)
        self.thread.daemon = True
        self.thread.start()

        # Load YOLO model (offline if available)
        if os.path.exists(MODEL_PATH):
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
            self.get_logger().info(f"Loaded YOLO model from {MODEL_PATH}")
        else:
            self.get_logger().info("Downloading YOLOv5n pretrained model...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            self.model.save(MODEL_PATH)
            self.get_logger().info(f"Downloaded and saved YOLO model to {MODEL_PATH}")
        self.model.to(self.device)
        self.model.eval()

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
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                # Convert to torch tensor
                img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                results = self.model(img_rgb, size=640).to(self.device)  # you can adjust size
                self.publish_results(results, msg.header)
            except Exception as e:
                self.get_logger().error(f"YOLO processing error: {e}")

    def publish_results(self, results, header):
        # Publish Detection2DArray (bounding boxes, classes, confidence)
        det_array = Detection2DArray()
        det_array.header = header

        # YOLO results as a pandas dataframe: xmin, ymin, xmax, ymax, confidence, class, name
        df = results.pandas().xyxy[0]

        for _, row in df.iterrows():
            det = Detection2D()

            # Bounding box center + size
            det.bbox.center.x = float((row.xmin + row.xmax) / 2.0)
            det.bbox.center.y = float((row.ymin + row.ymax) / 2.0)
            det.bbox.size_x = float(row.xmax - row.xmin)
            det.bbox.size_y = float(row.ymax - row.ymin)

            # Class + confidence
            hyp = ObjectHypothesisWithPose()
            hyp.id = int(row['class'])          # YOLO class ID
            hyp.score = float(row['confidence']) # Confidence score
            det.results.append(hyp)

            det_array.detections.append(det)

        # Publish only detections
        self.detection_pub.publish(det_array)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
