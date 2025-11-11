
"""
Test script to verify FaceNet model is producing diverse embeddings.
Tests with different synthetic inputs to see if embeddings are diverse.
"""

import numpy as np
import onnxruntime as ort
import cv2
import urllib.request
import os

CURRENT_MODEL_PATH = "/ball-e/ros2_ws/src/perception_pkg/perception_pkg/models/facenet.onnx"

# Download alternative FaceNet model from ONNX Model Zoo or other source
ALTERNATIVE_MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"
ALTERNATIVE_MODEL_PATH = "/tmp/arcface_test.onnx"

print("Downloading alternative face recognition model...")
if not os.path.exists(ALTERNATIVE_MODEL_PATH):
    try:
        urllib.request.urlretrieve(ALTERNATIVE_MODEL_URL, ALTERNATIVE_MODEL_PATH)
        print(f"✓ Downloaded alternative model to {ALTERNATIVE_MODEL_PATH}")
    except Exception as e:
        print(f"✗ Failed to download alternative model: {e}")
        print("Will test with current model only")
        ALTERNATIVE_MODEL_PATH = None
else:
    print(f"✓ Alternative model already exists at {ALTERNATIVE_MODEL_PATH}")

MODEL_PATH = CURRENT_MODEL_PATH

def preprocess_face_neg1_to_1(face: np.ndarray, input_size: int) -> np.ndarray:
    """Preprocess face with [-1, 1] normalization (current implementation)"""
    resized = cv2.resize(face, (input_size, input_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) - 127.5) / 128.0
    chw = np.transpose(normalized, (2, 0, 1))
    batch = np.expand_dims(chw, axis=0)
    return batch

def preprocess_face_0_to_1(face: np.ndarray, input_size: int) -> np.ndarray:
    """Preprocess face with [0, 1] normalization (alternative)"""
    resized = cv2.resize(face, (input_size, input_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    batch = np.expand_dims(chw, axis=0)
    return batch

def test_model(model_path, model_name, test_images, image_names):
    """Test a model with different preprocessing methods"""
    print("\n" + "="*80)
    print(f"TESTING MODEL: {model_name}")
    print("="*80)

    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape

        print(f"Model input: name={input_name}, shape={input_shape}")
        print(f"Model output: name={output_name}, shape={session.get_outputs()[0].shape}")

        model_input_size = input_shape[2]  # Assuming [batch, channels, height, width]
        print(f"Model input size: {model_input_size}x{model_input_size}")

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    # Test with both preprocessing methods
    for preprocess_name, preprocess_func in [
        ("[-1, 1] normalization", preprocess_face_neg1_to_1),
        ("[0, 1] normalization", preprocess_face_0_to_1)
    ]:
        print("\n" + "-"*80)
        print(f"Preprocessing: {preprocess_name}")
        print("-"*80)

        embeddings = []

        for i, img in enumerate(test_images):
            try:
                preprocessed = preprocess_func(img, model_input_size)
                embedding = session.run([output_name], {input_name: preprocessed})[0]

                # L2 normalize
                norm = np.linalg.norm(embedding)
                embedding_normalized = embedding / norm
                embedding_flat = embedding_normalized.flatten()
                embeddings.append(embedding_flat)

                print(f"[{i}] {image_names[i][:40]:40s}: embedding_dim={len(embedding_flat)}, norm={np.linalg.norm(embedding_flat):.6f}")

            except Exception as e:
                print(f"[{i}] {image_names[i]}: ERROR - {e}")
                embeddings.append(None)

        # Compute similarities
        print(f"\nSimilarity scores ({preprocess_name}):")
        for i in range(len(embeddings)):
            if embeddings[i] is None:
                continue
            for j in range(i+1, len(embeddings)):
                if embeddings[j] is None:
                    continue
                similarity = np.dot(embeddings[i], embeddings[j])
                status = "⚠️  HIGH!" if similarity > 0.9 else ("✓ Good" if similarity < 0.5 else "  OK")
                print(f"  [{i}] vs [{j}]: {similarity:.6f} {status}")

# Load test face images
import glob

test_faces_dir = "/ball-e/test_faces"
image_paths = sorted(glob.glob(os.path.join(test_faces_dir, "*.jpg")) + glob.glob(os.path.join(test_faces_dir, "*.jpeg")))

if not image_paths:
    print(f"ERROR: No images found in {test_faces_dir}")
    exit(1)

print("\n" + "="*80)
print(f"Loading {len(image_paths)} test face images from {test_faces_dir}")
print("="*80)

test_images = []
image_names = []
for i, img_path in enumerate(image_paths):
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Failed to load {img_path}")
        continue
    test_images.append(img)
    image_names.append(os.path.basename(img_path))
    print(f"  [{i}] {os.path.basename(img_path)} - shape={img.shape}")

# Test current model
test_model(CURRENT_MODEL_PATH, "CURRENT MODEL (facenet.onnx)", test_images, image_names)

# Test alternative model if available
if ALTERNATIVE_MODEL_PATH and os.path.exists(ALTERNATIVE_MODEL_PATH):
    test_model(ALTERNATIVE_MODEL_PATH, "ALTERNATIVE MODEL (ArcFace)", test_images, image_names)

print("\n" + "="*80)
print("SUMMARY:")
print("- If current model shows high similarities (>0.9) for different people: Model is broken")
print("- If alternative model shows low similarities (<0.5): Use the alternative model")
print("- Check which preprocessing method ([0,1] vs [-1,1]) works best for each model")
print("="*80)
