# IdentificationRequest.msg
# Message published when a person requires identification

std_msgs/Header header  # Timestamp from original frame
int32 track_id
int32 bbox_x
int32 bbox_y
int32 bbox_w
int32 bbox_h
