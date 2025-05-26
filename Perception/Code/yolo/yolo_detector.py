from ultralytics import YOLO
import torch
from Perception.Code.yolo.utils import load_config, load_class_names_from_yaml


class YOLODetector:
    def __init__(self, config_path='Perception\Code\yolo\yolo_config.yaml'):
        config = load_config(config_path)

        self.device = config.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.allowed_classes = config.get('allowed_classes', [0])  # Default to person

        self.model = YOLO(config['model_path']).to(self.device)
        self.class_names = self.model.names

    def detect(self, frame):
        results = self.model(frame, device=self.device)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if self.allowed_classes is None or cls_id in self.allowed_classes:
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_id': cls_id,
                    'label': self.class_names[cls_id],
                    'confidence': conf
                })

        return detections
