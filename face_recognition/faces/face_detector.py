
import insightface
import numpy as np
import cv2
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, det_thresh=0.6):
        self.detector = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])  # GPU
        self.detector.prepare(ctx_id=0)
        self.det_thresh = det_thresh

    def detect_and_embed(self, rgb_crop):
        faces = self.detector.get(rgb_crop)
        if faces is None or len(faces) == 0:
            return None, None

        face = faces[0]
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        return bbox, embedding