import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognizer:
    def __init__(self, face_database, threshold=0.5):
        self.db = face_database
        self.threshold = threshold

    def recognize(self, embedding):
        known_faces = self.db.get_all()
        if not known_faces:
            return None, None

        names = list(known_faces.keys())
        embeddings = np.array(list(known_faces.values()))

        similarities = cosine_similarity([embedding], embeddings)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score > self.threshold:
            return names[best_idx], best_score
        return None, None
