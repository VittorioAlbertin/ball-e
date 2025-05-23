import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class FaceDatabase:
    def __init__(self, path="known_faces.pkl", threshold=0.5):
        base_dir = os.path.dirname(os.path.abspath(__file__))  # folder of face_database.py
        self.path = os.path.join(base_dir, path)               # full path to known_faces.pkl
        self.threshold = threshold
        self.faces = self._load_faces()


    def _load_faces(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                return pickle.load(f)
        return {}

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.faces, f)

    def add_face(self, name, embedding):
        self.faces[name] = embedding
        self.save()

    def remove_face(self, name):
        if name in self.faces:
            del self.faces[name]
            self.save()
            return True
        return False


    def recognize(self, embedding):
        if not self.faces:
            return None, None

        names = list(self.faces.keys())
        embeddings = np.array(list(self.faces.values()))
        similarities = cosine_similarity([embedding], embeddings)[0]
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]

        if best_score > self.threshold:
            return names[best_match_idx], best_score
        return None, None