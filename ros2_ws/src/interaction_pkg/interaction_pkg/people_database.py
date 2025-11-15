import sqlite3
import numpy as np
from datetime import datetime
import json
from typing import Optional, List, Dict, Tuple
import pickle


class PeopleDatabase:
    """
    SQLite-based database for managing people interacting with the robot.
    Stores names, face embeddings, and interaction metadata.
    """
    
    def __init__(self, db_path: str = "/ball-e/ros2_ws/robot_data/people.db"):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to SQLite database file. 
                     Mount a Docker volume here for persistence.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        self._create_tables()
    
    def _create_tables(self):
        """Create the people table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                face_embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                voice_embedding BLOB,
                voice_embedding_dim INTEGER,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                interaction_count INTEGER DEFAULT 1,
                preferences TEXT,
                notes TEXT
            )
        """)

        # Create table for multiple face embeddings (different poses)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                pose_type TEXT,
                quality_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
            )
        """)

        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_name ON people(name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_seen ON people(last_seen)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_face_embeddings_person
            ON face_embeddings(person_id)
        """)

        self.conn.commit()
    
    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Convert numpy array to bytes for storage."""
        return pickle.dumps(embedding)
    
    def _deserialize_embedding(self, blob: bytes) -> np.ndarray:
        """Convert bytes back to numpy array."""
        return pickle.loads(blob)
    
    def add_person(
        self,
        name: str,
        face_embedding: np.ndarray,
        preferences: Optional[Dict] = None,
        notes: Optional[str] = None,
        voice_embedding: Optional[np.ndarray] = None
    ) -> int:
        """
        Add a new person to the database.

        Args:
            name: Person's name
            face_embedding: Face embedding vector (numpy array)
            preferences: Optional dict of user preferences
            notes: Optional notes about the person
            voice_embedding: Optional voice embedding vector (numpy array)

        Returns:
            The ID of the newly created person record
        """
        cursor = self.conn.cursor()

        embedding_blob = self._serialize_embedding(face_embedding)
        preferences_json = json.dumps(preferences) if preferences else None

        # Handle voice embedding
        voice_blob = None
        voice_dim = None
        if voice_embedding is not None:
            voice_blob = self._serialize_embedding(voice_embedding)
            voice_dim = len(voice_embedding)

        cursor.execute("""
            INSERT INTO people (name, face_embedding, embedding_dim, voice_embedding, voice_embedding_dim, preferences, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, embedding_blob, len(face_embedding), voice_blob, voice_dim, preferences_json, notes))

        self.conn.commit()
        return cursor.lastrowid

    def add_face_embedding(
        self,
        person_id: int,
        embedding: np.ndarray,
        pose_type: str = 'unknown',
        quality_score: float = 1.0
    ) -> int:
        """
        Add a face embedding for a specific pose to the face_embeddings table.

        Args:
            person_id: Person's database ID
            embedding: Face embedding vector
            pose_type: Type of pose ('front', 'left', 'right', 'up', 'down')
            quality_score: Quality score of the embedding (0-1)

        Returns:
            The ID of the newly created embedding record
        """
        cursor = self.conn.cursor()
        embedding_blob = self._serialize_embedding(embedding)

        cursor.execute("""
            INSERT INTO face_embeddings (person_id, embedding, embedding_dim, pose_type, quality_score)
            VALUES (?, ?, ?, ?, ?)
        """, (person_id, embedding_blob, len(embedding), pose_type, quality_score))

        self.conn.commit()
        return cursor.lastrowid

    def get_face_embeddings_for_person(self, person_id: int) -> List[Dict]:
        """
        Get all face embeddings for a person (different poses).

        Args:
            person_id: Person's database ID

        Returns:
            List of dicts with embedding, pose_type, quality_score
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT embedding, pose_type, quality_score
            FROM face_embeddings
            WHERE person_id = ?
        """, (person_id,))
        rows = cursor.fetchall()

        return [{
            'embedding': self._deserialize_embedding(row['embedding']),
            'pose_type': row['pose_type'],
            'quality_score': row['quality_score']
        } for row in rows]

    def update_voice_embedding(self, person_id: int, voice_embedding: np.ndarray):
        """
        Update or add voice embedding for a person.

        Args:
            person_id: Person's database ID
            voice_embedding: Voice embedding vector
        """
        cursor = self.conn.cursor()
        voice_blob = self._serialize_embedding(voice_embedding)

        cursor.execute("""
            UPDATE people
            SET voice_embedding = ?, voice_embedding_dim = ?
            WHERE id = ?
        """, (voice_blob, len(voice_embedding), person_id))

        self.conn.commit()

    def get_all_voice_embeddings(self) -> List[Tuple[int, np.ndarray]]:
        """
        Get all voice embeddings for similarity comparison.

        Returns:
            List of tuples (person_id, voice_embedding)
            Only returns persons with voice embeddings
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, voice_embedding
            FROM people
            WHERE voice_embedding IS NOT NULL
        """)
        rows = cursor.fetchall()

        return [(row['id'], self._deserialize_embedding(row['voice_embedding']))
                for row in rows]

    def find_similar_voice(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.75,
        logger=None
    ) -> Optional[tuple]:
        """
        Find a person with similar voice embedding using cosine similarity.

        Args:
            query_embedding: Voice embedding to search for
            threshold: Similarity threshold (0-1), higher = more strict
            logger: Optional ROS logger for debug output

        Returns:
            Tuple of (person_dict, similarity_score) if match found, None otherwise
        """
        embeddings = self.get_all_voice_embeddings()

        if not embeddings:
            if logger:
                logger.info("No voice embeddings in database")
            return None

        best_match_id = None
        best_match_name = None
        best_similarity = -1

        # Normalize query embedding
        query_norm_value = np.linalg.norm(query_embedding)
        if abs(query_norm_value - 1.0) > 0.01:
            query_norm = query_embedding / query_norm_value
        else:
            query_norm = query_embedding

        all_similarities = []

        for person_id, stored_embedding in embeddings:
            # Normalize stored embedding
            stored_norm_value = np.linalg.norm(stored_embedding)
            if abs(stored_norm_value - 1.0) > 0.01:
                stored_norm = stored_embedding / stored_norm_value
            else:
                stored_norm = stored_embedding

            # Cosine similarity
            similarity = np.dot(query_norm, stored_norm)

            person = self.get_person_by_id(person_id)
            person_name = person['name'] if person else f"ID_{person_id}"
            all_similarities.append((person_name, similarity))

            if logger:
                logger.info(f"[VOICE MATCH] {person_name}: similarity={similarity:.6f}")

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id
                best_match_name = person_name

        if logger:
            logger.info(f"Voice matching results (threshold={threshold:.2f}):")
            for name, sim in sorted(all_similarities, key=lambda x: x[1], reverse=True):
                status = "✓ MATCH" if sim >= threshold else "✗ below threshold"
                logger.info(f"  {name}: {sim:.4f} {status}")

        if best_similarity >= threshold:
            return (self.get_person_by_id(best_match_id), best_similarity)
        else:
            return None

    def get_all_face_scores(
        self,
        query_embedding: np.ndarray,
        logger=None
    ) -> Dict[int, float]:
        """
        Get similarity scores for all persons (for Bayesian fusion).

        Args:
            query_embedding: Face embedding to compare
            logger: Optional logger

        Returns:
            Dict mapping person_id to similarity score
        """
        embeddings = self.get_all_embeddings()
        scores = {}

        if not embeddings:
            return scores

        # Normalize query
        query_norm_value = np.linalg.norm(query_embedding)
        if abs(query_norm_value - 1.0) > 0.01:
            query_norm = query_embedding / query_norm_value
        else:
            query_norm = query_embedding

        for person_id, stored_embedding in embeddings:
            stored_norm_value = np.linalg.norm(stored_embedding)
            if abs(stored_norm_value - 1.0) > 0.01:
                stored_norm = stored_embedding / stored_norm_value
            else:
                stored_norm = stored_embedding

            similarity = float(np.dot(query_norm, stored_norm))
            scores[person_id] = similarity

            if logger:
                person = self.get_person_by_id(person_id)
                name = person['name'] if person else f"ID_{person_id}"
                logger.debug(f"[FACE SCORE] {name}: {similarity:.4f}")

        return scores

    def get_all_voice_scores(
        self,
        query_embedding: np.ndarray,
        logger=None
    ) -> Dict[int, float]:
        """
        Get voice similarity scores for all persons (for Bayesian fusion).

        Args:
            query_embedding: Voice embedding to compare
            logger: Optional logger

        Returns:
            Dict mapping person_id to similarity score
        """
        embeddings = self.get_all_voice_embeddings()
        scores = {}

        if not embeddings:
            return scores

        # Normalize query
        query_norm_value = np.linalg.norm(query_embedding)
        if abs(query_norm_value - 1.0) > 0.01:
            query_norm = query_embedding / query_norm_value
        else:
            query_norm = query_embedding

        for person_id, stored_embedding in embeddings:
            stored_norm_value = np.linalg.norm(stored_embedding)
            if abs(stored_norm_value - 1.0) > 0.01:
                stored_norm = stored_embedding / stored_norm_value
            else:
                stored_norm = stored_embedding

            similarity = float(np.dot(query_norm, stored_norm))
            scores[person_id] = similarity

            if logger:
                person = self.get_person_by_id(person_id)
                name = person['name'] if person else f"ID_{person_id}"
                logger.debug(f"[VOICE SCORE] {name}: {similarity:.4f}")

        return scores
    
    def update_last_seen(self, person_id: int):
        """
        Update the last_seen timestamp and increment interaction count.
        Call this when you recognize someone.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE people 
            SET last_seen = CURRENT_TIMESTAMP,
                interaction_count = interaction_count + 1
            WHERE id = ?
        """, (person_id,))
        self.conn.commit()
    
    def get_person_by_id(self, person_id: int) -> Optional[Dict]:
        """Get person data by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM people WHERE id = ?", (person_id,))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_dict(row)
        return None
    
    def get_person_by_name(self, name: str) -> Optional[Dict]:
        """Get person data by name."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM people WHERE name = ?", (name,))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_dict(row)
        return None
    
    def get_all_people(self) -> List[Dict]:
        """Get all people in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM people ORDER BY last_seen DESC")
        rows = cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    def get_all_embeddings(self) -> List[Tuple[int, np.ndarray]]:
        """
        Get all face embeddings for similarity comparison.
        
        Returns:
            List of tuples (person_id, embedding)
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, face_embedding FROM people")
        rows = cursor.fetchall()
        
        return [(row['id'], self._deserialize_embedding(row['face_embedding'])) 
                for row in rows]
    
    def find_similar_face(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.6,
        logger=None
    ) -> Optional[tuple]:
        """
        Find a person with similar face embedding using cosine similarity.

        Args:
            query_embedding: Face embedding to search for (should already be L2 normalized)
            threshold: Similarity threshold (0-1), higher = more strict
            logger: Optional ROS logger for debug output

        Returns:
            Tuple of (person_dict, similarity_score) if match found, None otherwise
        """
        embeddings = self.get_all_embeddings()

        if not embeddings:
            if logger:
                logger.info("No embeddings in database")
            return None

        best_match_id = None
        best_match_name = None
        best_similarity = -1

        # Check if query embedding is normalized (embeddings from FaceNet should already be L2 normalized)
        query_norm_value = np.linalg.norm(query_embedding)
        if logger:
            logger.info(f"Query embedding norm: {query_norm_value:.6f} (should be ~1.0 if already normalized)")

        # Only normalize if not already normalized (to avoid numerical errors from double normalization)
        if abs(query_norm_value - 1.0) > 0.01:
            if logger:
                logger.warning(f"Query embedding not normalized (norm={query_norm_value:.6f}), normalizing now")
            query_norm = query_embedding / query_norm_value
        else:
            query_norm = query_embedding

        # Track all similarities for logging
        all_similarities = []

        for person_id, stored_embedding in embeddings:
            # Check if stored embedding is normalized
            stored_norm_value = np.linalg.norm(stored_embedding)

            # Only normalize if not already normalized
            if abs(stored_norm_value - 1.0) > 0.01:
                if logger:
                    logger.warning(f"Stored embedding for person {person_id} not normalized (norm={stored_norm_value:.6f}), normalizing now")
                stored_norm = stored_embedding / stored_norm_value
            else:
                stored_norm = stored_embedding

            # Cosine similarity via dot product (for normalized vectors)
            similarity = np.dot(query_norm, stored_norm)

            # Get person name for logging
            person = self.get_person_by_id(person_id)
            person_name = person['name'] if person else f"ID_{person_id}"
            all_similarities.append((person_name, similarity))

            # DEBUG: Log embedding comparison details
            if logger:
                logger.info(f"[MATCH] {person_name}: query_norm={query_norm_value:.6f}, stored_norm={stored_norm_value:.6f}, similarity={similarity:.6f}")
                logger.info(f"[MATCH] {person_name}: query first 5: {query_norm[:5]}, stored first 5: {stored_norm[:5]}")

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id
                best_match_name = person_name

        # Log all similarities for debugging
        if logger:
            logger.info(f"Face matching results (threshold={threshold:.2f}):")
            for name, sim in sorted(all_similarities, key=lambda x: x[1], reverse=True):
                status = "✓ MATCH" if sim >= threshold else "✗ below threshold"
                logger.info(f"  {name}: {sim:.4f} {status}")
            logger.info(f"Best match: {best_match_name} with similarity {best_similarity:.4f}")

        if best_similarity >= threshold:
            if logger:
                logger.info(f"Accepting match: {best_match_name} (similarity {best_similarity:.4f} >= threshold {threshold:.2f})")
            return (self.get_person_by_id(best_match_id), best_similarity)
        else:
            if logger:
                logger.info(f"No match found. Best similarity {best_similarity:.4f} < threshold {threshold:.2f}")
            return None
    
    def update_preferences(self, person_id: int, preferences: Dict):
        """Update user preferences."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE people 
            SET preferences = ?
            WHERE id = ?
        """, (json.dumps(preferences), person_id))
        self.conn.commit()
    
    def update_notes(self, person_id: int, notes: str):
        """Update notes about a person."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE people 
            SET notes = ?
            WHERE id = ?
        """, (notes, person_id))
        self.conn.commit()
    
    def delete_person(self, person_id: int):
        """Delete a person from the database."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM people WHERE id = ?", (person_id,))
        self.conn.commit()
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert database row to dictionary."""
        result = {
            'id': row['id'],
            'name': row['name'],
            'face_embedding': self._deserialize_embedding(row['face_embedding']),
            'embedding_dim': row['embedding_dim'],
            'last_seen': row['last_seen'],
            'created_at': row['created_at'],
            'interaction_count': row['interaction_count'],
            'preferences': json.loads(row['preferences']) if row['preferences'] else None,
            'notes': row['notes']
        }

        # Add voice embedding if available (handle backward compatibility)
        try:
            if row['voice_embedding'] is not None:
                result['voice_embedding'] = self._deserialize_embedding(row['voice_embedding'])
                result['voice_embedding_dim'] = row['voice_embedding_dim']
            else:
                result['voice_embedding'] = None
                result['voice_embedding_dim'] = None
        except (IndexError, KeyError):
            # Old database schema without voice columns
            result['voice_embedding'] = None
            result['voice_embedding_dim'] = None

        return result

    def migrate_schema(self):
        """
        Migrate database schema to add new columns if they don't exist.
        Call this once when upgrading from old schema.
        """
        cursor = self.conn.cursor()

        # Check if voice_embedding column exists
        cursor.execute("PRAGMA table_info(people)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'voice_embedding' not in columns:
            cursor.execute("ALTER TABLE people ADD COLUMN voice_embedding BLOB")
            cursor.execute("ALTER TABLE people ADD COLUMN voice_embedding_dim INTEGER")

        # Create face_embeddings table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                pose_type TEXT,
                quality_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_face_embeddings_person
            ON face_embeddings(person_id)
        """)

        self.conn.commit()
    
    def close(self):
        """Close the database connection."""
        self.conn.close()


# Example usage in a ROS2 node
if __name__ == "__main__":
    # Initialize database
    db = PeopleDatabase("/ball-e/ros2_ws/robot_data/people.db")
    
    # Example: Add a new person (simulate face embedding with random vector)
    fake_embedding = np.random.rand(512)  # Typical face embedding size
    person_id = db.add_person(
        name="John Doe",
        face_embedding=fake_embedding,
        preferences={"language": "en", "greeting_style": "formal"},
        notes="Met at the lab entrance"
    )
    print(f"Added person with ID: {person_id}")
    
    # Example: Find similar face
    query_embedding = fake_embedding + np.random.rand(512) * 0.1  # Similar but not identical
    match = db.find_similar_face(query_embedding, threshold=0.6)
    if match:
        print(f"Found match: {match['name']}")
        db.update_last_seen(match['id'])
    else:
        print("No match found")
    
    # Example: Get all people
    all_people = db.get_all_people()
    for person in all_people:
        print(f"{person['name']}: seen {person['interaction_count']} times, "
              f"last seen {person['last_seen']}")
    
    db.close()