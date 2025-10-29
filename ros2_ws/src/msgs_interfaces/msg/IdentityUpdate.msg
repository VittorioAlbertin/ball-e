# IdentityUpdate.msg
# Message published when a person's identity is updated

std_msgs/Header header
int32 track_id
string identity  # Person name or 'unknown'
float32 confidence
int32 person_id  # Database ID or -1
