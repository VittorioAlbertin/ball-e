import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, det_thresh=0.6, providers=['CUDAExecutionProvider']):
        self.detector = FaceAnalysis(name='buffalo_l', providers=providers)
        self.detector.prepare(ctx_id=0)
        self.det_thresh = det_thresh

    def detect_faces(self, rgb_image):
        faces = self.detector.get(rgb_image)
        return [face for face in faces if face.det_score > self.det_thresh]

    def extract_main_face(self, rgb_image):
        faces = self.detect_faces(rgb_image)
        if not faces:
            return None, None
        face = faces[0]
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        return bbox, embedding
