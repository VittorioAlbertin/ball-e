import cv2
import numpy as np
from Perception.Code.face_detection.face_detector import FaceDetector
from Perception.Code.face_recognition.face_database import FaceDatabase
from Perception.Code.face_recognition.recognizer import FaceRecognizer
from Perception.Code.yolo.yolo_detector import YOLODetector


class FaceProcessingPipeline:
    def __init__(self, recognition_threshold=0.5):
        # Initialize components
        self.face_detector = FaceDetector()
        self.face_db = FaceDatabase()
        self.recognizer = FaceRecognizer(self.face_db, threshold=recognition_threshold)
        self.yolo = YOLODetector()

    def process_frame(self, frame):
        """Process a BGR frame and return a list of recognition results."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.yolo.detect(frame)

        if not detections:
            return [], "No person detected"

        results = []

        for det in detections:
            if det["class_id"] == 0:  # Only process person detections
                x1, y1, x2, y2 = det["bbox"]
                person_crop = rgb[y1:y2, x1:x2]

                # Detect face and extract embedding within person's ROI
                bbox_face, embedding = self.face_detector.extract_main_face(person_crop)

                if bbox_face is None or embedding is None:
                    continue  # No face in this person ROI

                # Adjust face bbox to original frame coordinates
                x_face, y_face, x2_face, y2_face = bbox_face
                full_frame_bbox = (x1 + x_face, y1 + y_face, x1 + x2_face, y1 + y2_face)

                # Recognize face
                identity, score = self.recognizer.recognize(embedding)

                results.append({
                    "bbox": full_frame_bbox,
                    "embedding": embedding,
                    "identity": identity,
                    "confidence": score
                })

        return results, None if results else "No faces recognized"
