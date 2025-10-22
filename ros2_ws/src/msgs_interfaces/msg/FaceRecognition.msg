# FaceRecognition.msg
# Message for face recognition results

std_msgs/Header header

# Face embedding vector
float32[] face_embedding

# Bounding box in original frame coordinates
float32 bbox_center_x
float32 bbox_center_y
float32 bbox_size_x
float32 bbox_size_y

# Detection confidence
float32 confidence

# Recognition results (from RecogniseFace service)
bool found
int32 person_id
string name
string last_seen
int32 interaction_count
string preferences_json
string notes
string message