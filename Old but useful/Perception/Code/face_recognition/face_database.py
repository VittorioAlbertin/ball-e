import os
import pickle
import numpy as np

class FaceDatabase:
    def __init__(self, db_path="known_faces.pkl"):
        self.path = os.path.join(os.path.dirname(__file__), db_path)
        self.faces = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                return pickle.load(f)
        return {}

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.faces, f)

    def add(self, name, embedding):
        self.faces[name] = embedding
        self.save()

    def remove(self, name):
        if name in self.faces:
            del self.faces[name]
            self.save()
            return True
        return False

    def get_all(self):
        return self.faces
