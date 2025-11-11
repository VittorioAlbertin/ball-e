# PersonState.msg
# Represents the complete state of a tracked person in the world model

std_msgs/Header header

# Tracking information
int32 track_id

# Identity information (nullable - empty string means unidentified)
string identity
float32 identity_confidence

# Spatial information
float32 bbox_x
float32 bbox_y
float32 bbox_w
float32 bbox_h

# Temporal information
builtin_interfaces/Time first_seen
builtin_interfaces/Time last_seen

# Recognition control flags
bool requires_identification

# Tracking quality
float32 tracking_confidence
int32 frames_since_last_seen