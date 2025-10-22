# IdentityUpdate.msg
# Message for updating person identity after face recognition

std_msgs/Header header

# Track ID of the person
int32 track_id

# Identified name (empty string means unknown/no match)
string identity

# Recognition confidence (0.0 to 1.0)
float32 confidence

# Face embedding (for enrollment)
float32[] face_embedding

# Processing time in milliseconds
float32 processing_time_ms

# Face quality metrics
float32 face_size_pixels
bool face_quality_ok

# Recognition details
bool found_in_database
int32 person_id
string message
