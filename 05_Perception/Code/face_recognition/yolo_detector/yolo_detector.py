from ultralytics import YOLO
import torch

class YOLODetector:
    def __init__(self, model_name='yolov5s', device=None, allowed_classes=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_name).to(self.device)

        # Default to detecting only 'person' if no classes are specified
        self.allowed_classes = allowed_classes if allowed_classes is not None else [0]

        # COCO class names for labeling
        self.class_names = self.model.names

    def detect(self, frame):
        results = self.model(frame, device=self.device)[0]
        people = []
        others = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.allowed_classes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'class_id': cls_id,
                    'label': self.class_names[cls_id],
                    'confidence': conf
                }

                if cls_id == 0:
                    people.append(detection)
                else:
                    others.append(detection)

        return people, others