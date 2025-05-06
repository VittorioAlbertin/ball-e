import cv2
import pickle
from yolo_people_detection.detector import YOLODetector
from recognition.face_recognizer import FaceRecognizer
from recognition.face_database import FaceDatabase

def main():
    detector = YOLODetector(model_name='yolov5su.pt')
    recognizer = FaceRecognizer()
    face_database = FaceDatabase()
    cap = cv2.VideoCapture(0)

    # Load known faces from the file
    try:
        face_database._load_faces()
    except FileNotFoundError:
        print("No previously saved faces found, starting fresh.")
    
    print("Press 's' to save a new face. Press 'q' to quit.")
    
    unknown_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        people = detector.detect_people(frame)
        unknown_faces.clear()  # Clear previous frame's unknowns

        for person in people:
            x1, y1, x2, y2 = person['bbox']
            person_crop = frame[y1:y2, x1:x2]
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            face_bbox, embedding = recognizer.detect_and_embed(rgb_crop)
            if embedding is not None:
                fx1, fy1, fx2, fy2 = face_bbox
                abs_x1 = x1 + fx1
                abs_y1 = y1 + fy1
                abs_x2 = x1 + fx2
                abs_y2 = y1 + fy2

                cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (255, 0, 0), 2)

                name = face_database.recognize(embedding)
                if name:
                    cv2.putText(frame, name, (abs_x1, abs_y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (abs_x1, abs_y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cropped_face = frame[abs_y1:abs_y2, abs_x1:abs_x2]
                    if cropped_face.size != 0:
                        unknown_faces.append((embedding, cropped_face))


        cv2.imshow("BALL·E - Person & Face Detection", frame)
        key = cv2.waitKey(1)

        if key == ord("s"):
            for i, (embedding, cropped_face) in enumerate(unknown_faces):
                win_name = f"Face {i+1}"
                cv2.imshow(win_name, cropped_face)
                cv2.waitKey(1)

                name_input = input(f"Enter name for Face {i+1} (or leave blank to skip): ").strip()
                cv2.destroyWindow(win_name)

                if name_input:
                    face_database.add_face(name_input, embedding)
                    print(f"Saved embedding for {name_input}")
                else:
                    print("Skipped.")

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
