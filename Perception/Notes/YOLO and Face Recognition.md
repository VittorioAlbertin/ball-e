### Goals

1. **Modularization**: Separate reusable components for YOLO, face detection, face recognition, and database management.
    
2. **Reusability**: Allow the YOLO object detector to be used throughout the system, independent of face recognition.
    
3. **ROS Compatibility**: Organize modules in a way that maps easily to future ROS nodes/packages.
    
4. **Scalability**: Ensure it's easy to replace or upgrade detection and recognition models in the future.
    

---

### Proposed Folder Structure

```
05_Perception/
└── Code/
    ├── object_detection/
    │   └── yolo/
    │       ├── yolo_detector.py
    │       └── config.yaml

    ├── face_recognition/
    │   ├── detection/
    │   │   └── face_detector.py
    │   ├── recognition/
    │   │   ├── recognizer.py
    │   │   └── face_database.py
    │   └── pipeline.py

    ├── tests/
    │   ├── test_yolo.py
    │   ├── test_face_detection.py
    │   └── test_recognition_pipeline.py
```

---

### Component Roles

#### `object_detection/yolo/yolo_detector.py`

- Loads a YOLO model.
    
- Handles detection of all classes (e.g. person, chair, cup).
    
- Returns object class, confidence score, and bounding box.
    
- Can be used for general-purpose perception (navigation, object manipulation).
    

#### `face_recognition/detection/face_detector.py`

- Performs face detection using MTCNN, Haar cascades, RetinaFace, or YOLO filtered by class.
    
- Returns bounding boxes and cropped face images.
    

#### `face_recognition/recognition/recognizer.py`

- Loads the facial embedding model.
    
- Compares embeddings with stored database.
    
- Returns name and similarity score for recognized individuals.
    

#### `face_recognition/recognition/face_database.py`

- Manages known faces and embeddings.
    
- Supports adding, removing, updating, and listing known identities.
    
- Saves and loads data from disk (e.g., JSON or pickle).
    

#### `face_recognition/utils/preprocessing.py`

- Handles image normalization, face alignment, and other preparation steps.
    

#### `face_recognition/pipeline.py`

- Coordinates the full face recognition pipeline:
    
    1. Run face detection
        
    2. Preprocess faces
        
    3. Extract embeddings
        
    4. Compare with the database
        
    5. Return annotations and identities
        

---
	
### Future ROS Integration (Preview)

Each component can be turned into a standalone ROS node:
```
ball_e_perception/
├── yolo_detector_node.py
├── face_detector_node.py
├── face_recognizer_node.py
└── msg/
    ├── DetectedObject.msg
    ├── DetectedFace.msg
    └── RecognizedPerson.msg
```

---

