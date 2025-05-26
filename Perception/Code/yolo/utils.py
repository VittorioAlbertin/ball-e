import yaml
import cv2


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_class_names_from_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data["names"] if "names" in data else data


def draw_detections(image, detections):
    for det in detections:
        #print(det) # debug
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image