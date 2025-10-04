#!/bin/bash

# Start VNC server with resolution 1280x720
vncserver :1 -geometry 1280x720 -depth 24

# Keep container alive
tail -f /root/.vnc/*.log
