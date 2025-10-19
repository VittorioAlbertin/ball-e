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
    
    def __init__(self, db_path: str = "/ros2_ws/robot_data/people.db"):
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
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                interaction_count INTEGER DEFAULT 1,
                preferences TEXT,
                notes TEXT
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_name ON people(name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_seen ON people(last_seen)
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
        notes: Optional[str] = None
    ) -> int:
        """
        Add a new person to the database.
        
        Args:
            name: Person's name
            face_embedding: Face embedding vector (numpy array)
            preferences: Optional dict of user preferences
            notes: Optional notes about the person
            
        Returns:
            The ID of the newly created person record
        """
        cursor = self.conn.cursor()
        
        embedding_blob = self._serialize_embedding(face_embedding)
        preferences_json = json.dumps(preferences) if preferences else None
        
        cursor.execute("""
            INSERT INTO people (name, face_embedding, embedding_dim, preferences, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (name, embedding_blob, len(face_embedding), preferences_json, notes))
        
        self.conn.commit()
        return cursor.lastrowid
    
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
        threshold: float = 0.6
    ) -> Optional[Dict]:
        """
        Find a person with similar face embedding using cosine similarity.
        
        Args:
            query_embedding: Face embedding to search for
            threshold: Similarity threshold (0-1), higher = more strict
            
        Returns:
            Person dict if match found, None otherwise
        """
        embeddings = self.get_all_embeddings()
        
        if not embeddings:
            return None
        
        best_match_id = None
        best_similarity = -1
        
        # Normalize query embedding
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        for person_id, stored_embedding in embeddings:
            # Cosine similarity
            stored_norm = stored_embedding / np.linalg.norm(stored_embedding)
            similarity = np.dot(query_norm, stored_norm)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id
        
        if best_similarity >= threshold:
            return self.get_person_by_id(best_match_id)
        
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
        return {
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
    
    def close(self):
        """Close the database connection."""
        self.conn.close()


# Example usage in a ROS2 node
if __name__ == "__main__":
    # Initialize database
    db = PeopleDatabase("/app/data/people.db")
    
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