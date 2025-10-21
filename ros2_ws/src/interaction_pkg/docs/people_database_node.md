# People Database Node

## Overview
The people database node provides persistent storage and retrieval services for recognized people. It manages face embeddings, personal information, preferences, and interaction history using an SQLite database.

## Node Information
- **Package**: `interaction_pkg`
- **Executable**: `people_database_node`
- **Node Name**: `people_database_node`

## Services

### Provided Services

| Service | Type | Description |
|---------|------|-------------|
| `people_db/add_person` | `msgs_interfaces/AddPerson` | Add a new person to the database |
| `people_db/recognize_face` | `msgs_interfaces/RecognizeFace` | Find matching person by face embedding |
| `people_db/get_person` | `msgs_interfaces/GetPerson` | Retrieve person info by ID |
| `people_db/update_last_seen` | `msgs_interfaces/UpdateLastSeen` | Update last interaction timestamp |
| `people_db/update_preferences` | `msgs_interfaces/UpdatePreferences` | Update person's preferences |
| `people_db/get_all_people` | `msgs_interfaces/GetAllPeople` | List all people in database |
| `people_db/delete_person` | `msgs_interfaces/DeletePerson` | Remove person from database |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | string | `/ball-e/ros2_ws/robot_data/people.db` | SQLite database file path |

## Database Schema

### People Table
```sql
CREATE TABLE people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    face_embedding BLOB NOT NULL,      -- 512-dim numpy array
    last_seen TEXT,                     -- ISO timestamp
    interaction_count INTEGER DEFAULT 0,
    preferences_json TEXT,              -- JSON string
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
```

## Service Details

### 1. add_person

Add a new person to the database.

**Request:**
```
string name                    # Person's name (required)
float32[] face_embedding       # 512-dim face vector (required)
string preferences_json        # Optional JSON preferences
string notes                   # Optional notes
```

**Response:**
```
bool success                   # True if added successfully
string message                 # Status/error message
int32 person_id                # Database ID of new person
```

**Example:**
```bash
ros2 service call /people_db/add_person msgs_interfaces/srv/AddPerson \
  "{name: 'John Doe', face_embedding: [...], notes: 'Friend from work'}"
```

### 2. recognize_face

Find the best matching person for a given face embedding.

**Request:**
```
float32[] face_embedding       # 512-dim face vector to match
float32 threshold              # Similarity threshold (0.0-1.0)
```

**Response:**
```
bool found                     # True if match found
int32 person_id                # Database ID (-1 if not found)
string name                    # Person's name
string last_seen               # Last interaction timestamp
int32 interaction_count        # Number of times seen
string preferences_json        # User preferences
string notes                   # Additional notes
string message                 # Status message
float32 similarity             # Cosine similarity score
```

**Matching Algorithm:**
- Computes cosine similarity between input and all stored embeddings
- Returns best match if similarity > threshold
- Updates `last_seen` and `interaction_count` on match

**Example:**
```bash
ros2 service call /people_db/recognize_face msgs_interfaces/srv/RecognizeFace \
  "{face_embedding: [...], threshold: 0.6}"
```

### 3. get_person

Retrieve person information by ID.

**Request:**
```
int32 person_id                # Database ID
```

**Response:**
```
bool success
string name
string last_seen
int32 interaction_count
string preferences_json
string notes
string message
```

### 4. update_last_seen

Update a person's last interaction timestamp.

**Request:**
```
int32 person_id                # Database ID
```

**Response:**
```
bool success
string message
```

### 5. update_preferences

Update a person's preferences (stored as JSON).

**Request:**
```
int32 person_id                # Database ID
string preferences_json        # JSON string
```

**Response:**
```
bool success
string message
```

**Preferences Example:**
```json
{
  "favorite_color": "blue",
  "coffee_preference": "cappuccino",
  "language": "en",
  "accessibility": {
    "needs_large_text": false,
    "prefers_audio": true
  }
}
```

### 6. get_all_people

List all people in the database.

**Request:**
```
# Empty
```

**Response:**
```
int32[] person_ids             # List of database IDs
string[] names                 # List of names (same order)
string message
```

### 7. delete_person

Remove a person from the database (PERMANENT).

**Request:**
```
int32 person_id                # Database ID to delete
```

**Response:**
```
bool success
string message
```

## Database Operations

### Face Embedding Storage
- Embeddings stored as binary BLOB (numpy array)
- Serialized using `tobytes()`
- Deserialized using `np.frombuffer()`
- Normalized to unit vectors for cosine similarity

### Similarity Calculation
```python
# Cosine similarity
similarity = np.dot(embedding1, embedding2)
# Both vectors are normalized, so no need to divide by norms
```

### Thread Safety
- SQLite handles concurrent reads
- Writes are serialized automatically
- No explicit locking required

## Usage Examples

### Python Client

```python
import rclpy
from rclpy.node import Node
from msgs_interfaces.srv import AddPerson, RecognizeFace
import numpy as np

class DatabaseClient(Node):
    def __init__(self):
        super().__init__('db_client')

        # Create service clients
        self.add_client = self.create_client(AddPerson, 'people_db/add_person')
        self.recognize_client = self.create_client(RecognizeFace, 'people_db/recognize_face')

        # Wait for services
        self.add_client.wait_for_service()
        self.recognize_client.wait_for_service()

    def add_person(self, name, embedding, notes=''):
        request = AddPerson.Request()
        request.name = name
        request.face_embedding = embedding.tolist()
        request.notes = notes

        future = self.add_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result().success:
            print(f"Added {name} with ID: {future.result().person_id}")
        else:
            print(f"Failed: {future.result().message}")

    def recognize_face(self, embedding, threshold=0.6):
        request = RecognizeFace.Request()
        request.face_embedding = embedding.tolist()
        request.threshold = threshold

        future = self.recognize_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result.found:
            print(f"Recognized: {result.name} (ID: {result.person_id})")
            print(f"Similarity: {result.similarity:.3f}")
            print(f"Seen {result.interaction_count} times")
        else:
            print("Not recognized")
```

## Backup and Recovery

### Manual Backup
```bash
# Backup database
cp /ball-e/ros2_ws/robot_data/people.db \
   /ball-e/ros2_ws/robot_data/people_backup_$(date +%Y%m%d).db

# Restore from backup
cp /ball-e/ros2_ws/robot_data/people_backup_20250101.db \
   /ball-e/ros2_ws/robot_data/people.db
```

### Export to JSON
```python
# Export all people (for migration/backup)
response = get_all_people_client.call(GetAllPeople.Request())
people_data = []
for person_id in response.person_ids:
    person_info = get_person_client.call(GetPerson.Request(person_id=person_id))
    people_data.append({
        'id': person_id,
        'name': person_info.name,
        'notes': person_info.notes,
        # Note: embeddings not exported for privacy
    })

with open('people_export.json', 'w') as f:
    json.dump(people_data, f, indent=2)
```

## Performance

### Query Speed
- **Add Person**: ~1-5ms
- **Recognize Face**: ~10-50ms (depends on database size)
  - Linear search through all embeddings
  - ~0.1ms per comparison
  - Consider indexing for >1000 people
- **Get Person**: ~1ms
- **Update**: ~1-5ms

### Scalability
- **Current**: Linear search, suitable for <1000 people
- **Future**: Consider FAISS/Annoy for >1000 people

## Security & Privacy

### Data Protection
- Face embeddings are irreversible (cannot reconstruct face)
- Database stored locally only
- No network transmission (all local services)
- Consider encryption for production use

### GDPR Compliance
- Implement data retention policies
- Provide delete functionality (already included)
- Consider consent tracking
- Audit trail for data access

## Troubleshooting

### Database Locked
```
Error: database is locked
```
**Solution**: Ensure only one database node is running

### Corrupted Database
```
Error: database disk image is malformed
```
**Solution**: Restore from backup or delete and recreate

### Recognition Not Working
1. Check embeddings are normalized
2. Verify threshold is appropriate (0.5-0.7)
3. Ensure same embedding model is used
4. Check embedding dimensionality (512)

### Service Not Available
```
Service not available: /people_db/add_person
```
**Solution**: Ensure `people_database_node` is running

## Dependencies
- `sqlite3` (Python standard library)
- `numpy`
- `msgs_interfaces` (custom services)

## Notes
- Database created automatically on first run
- Embeddings must be normalized for accurate recognition
- Cosine similarity used (not Euclidean distance)
- Service-based architecture (other nodes call services)
- No pub/sub topics (all service-based)
