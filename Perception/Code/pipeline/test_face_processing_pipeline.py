import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import time
import cv2
from face_processing_pipeline import FaceProcessingPipeline

# Initialize pipeline
pipeline = FaceProcessingPipeline(recognition_threshold=0.5)

# Open default webcam (0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

unknown_faces = []
prev_time = time.time()

while True:
    current_time = time.time()
    delta_time = max(current_time - prev_time, 1e-6)
    fps = 1 / delta_time
    prev_time = current_time

    ret, frame = cap.read()
    if not ret:
        break

    unknown_faces.clear()
    results, error = pipeline.process_frame(frame)

    if results:
        for result in results:
            x1, y1, x2, y2 = result["bbox"]
            identity = result["identity"] or "Unknown"
            conf = result["confidence"]
            embedding = result["embedding"]
            if identity != "Unknown":
                    label = f"{identity} ({conf:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
               cv2.putText(frame, "Unknown", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cropped_face = frame[y1:y2, x1:x2]
            if cropped_face.size != 0:
                unknown_faces.append((embedding, cropped_face))

    elif error:
        cv2.putText(frame, error, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("s"):
        for i, (embedding, cropped_face) in enumerate(unknown_faces):
            win_name = f"Face {i+1}"
            cv2.imshow(win_name, cropped_face)
            cv2.waitKey(1)

            name_input = input(f"Enter name for Face {i+1} (or leave blank to skip): ").strip()
            cv2.destroyWindow(win_name)

            if name_input:
                pipeline.face_db.add(name_input, embedding)
                print(f"Saved embedding for {name_input}")
            else:
                print("Skipped.")
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()