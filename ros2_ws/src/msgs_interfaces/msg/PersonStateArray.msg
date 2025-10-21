# PersonStateArray.msg
# Array of all currently tracked persons in the world model

std_msgs/Header header

# Array of person states
PersonState[] persons

# Summary statistics
int32 total_tracked
int32 identified_count
int32 unidentified_count
int32 pending_identification_count
