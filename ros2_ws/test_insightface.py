"""
Test script to verify InsightFace buffalo_l model produces diverse embeddings.
This is the model you remember using that worked well.
"""

import numpy as np
import cv2
import glob
import os

print("="*80)
print("Installing/Checking InsightFace...")
print("="*80)

try:
    from insightface.app import FaceAnalysis
    print("✓ InsightFace is already installed")
except ImportError:
    print("✗ InsightFace not installed. Installing now...")
    import subprocess
    subprocess.check_call(['pip3', 'install', 'insightface', 'onnxruntime'])
    from insightface.app import FaceAnalysis
    print("✓ InsightFace installed successfully")

# Initialize FaceAnalysis with buffalo_l model
print("\n" + "="*80)
print("Initializing InsightFace buffalo_l model...")
print("="*80)

# Use CPU provider (or CUDA if available)
providers = ['CPUExecutionProvider']

try:
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✓ InsightFace buffalo_l model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    print("\nTrying to download model...")
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✓ Model downloaded and loaded successfully")

# Load test face images
print("\n" + "="*80)
print("Loading test face images...")
print("="*80)

test_faces_dir = "/ball-e/test_faces"
image_paths = sorted(glob.glob(os.path.join(test_faces_dir, "*.jpg")) + glob.glob(os.path.join(test_faces_dir, "*.jpeg")))

if not image_paths:
    print(f"ERROR: No images found in {test_faces_dir}")
    exit(1)

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

# Extract embeddings using InsightFace
print("\n" + "="*80)
print("Extracting face embeddings with InsightFace...")
print("="*80)

embeddings = []

for i, (img, name) in enumerate(zip(test_images, image_names)):
    print(f"\n[{i}] {name}:")

    # Detect faces and extract embeddings
    faces = app.get(img)

    if len(faces) == 0:
        print("  ✗ No face detected")
        embeddings.append(None)
        continue

    if len(faces) > 1:
        print(f"  ⚠️  Multiple faces detected ({len(faces)}), using first one")

    # Get the first face's embedding
    face = faces[0]
    embedding = face.embedding

    # Normalize embedding
    embedding_norm = embedding / np.linalg.norm(embedding)
    embeddings.append(embedding_norm)

    print(f"  ✓ Face detected")
    print(f"  Embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.6f}")
    print(f"  Normalized: norm={np.linalg.norm(embedding_norm):.6f}")
    print(f"  Stats: min={embedding_norm.min():.6f}, max={embedding_norm.max():.6f}, mean={embedding_norm.mean():.6f}, std={embedding_norm.std():.6f}")
    print(f"  First 10 values: {embedding_norm[:10]}")

# Compute similarities
print("\n" + "="*80)
print("Computing similarity scores between face pairs...")
print("="*80)

for i in range(len(embeddings)):
    if embeddings[i] is None:
        continue
    for j in range(i+1, len(embeddings)):
        if embeddings[j] is None:
            continue

        similarity = np.dot(embeddings[i], embeddings[j])

        # Determine status
        if similarity > 0.9:
            status = "⚠️  HIGH - PROBLEM!"
        elif similarity > 0.6:
            status = "⚠️  MODERATE (might be same person)"
        elif similarity > 0.4:
            status = "✓ OK (different people)"
        else:
            status = "✓ GOOD (very different)"

        print(f"[{i}] {image_names[i][:35]:35s} vs [{j}] {image_names[j][:35]:35s}: {similarity:.6f} {status}")

print("\n" + "="*80)
print("ANALYSIS:")
print("Expected for DIFFERENT people: similarity < 0.6")
print("Expected for SAME person: similarity > 0.7")
print()
print("If InsightFace shows good results (low similarity for different people):")
print("  → We should replace the current FaceNet model with InsightFace")
print("="*80)
