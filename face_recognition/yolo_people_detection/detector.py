from ultralytics import YOLO
import torch

class YOLODetector:
    def __init__(self, model_name='yolov5s', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_name).to(self.device)

    def detect_people(self, frame):
        results = self.model(frame, device=self.device)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0:  # 'person' class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })
        return detections