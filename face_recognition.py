import cv2
import numpy as np
import onnxruntime as ort
from numpy.linalg import norm

# Load ArcFace ONNX model
model_path = 'model/arcface.onnx'
session = ort.InferenceSession(model_path)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

# Function to preprocess face and generate embedding using ArcFace
def get_face_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    face_pixels = cv2.resize(face_pixels, (112, 112))  # ArcFace input size
    face_pixels = np.expand_dims(face_pixels, axis=0)
    face_pixels = np.transpose(face_pixels, (0, 3, 1, 2))  # Change to NCHW format
    face_pixels = (face_pixels - 127.5) / 128.0  # Normalize as per ArcFace requirements

    # Generate embedding
    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: face_pixels})[0]
    return embedding[0]

# Function to process detected faces and return embeddings
def process_face(image, face_coordinates):
    (x, y, w, h) = face_coordinates
    face = image[y:y+h, x:x+w]
    embedding = get_face_embedding(face)
    return embedding

# Function to compare embeddings (face recognition)
def is_match(embedding1, embedding2, threshold=0.8):  # Adjusted threshold
    # Normalize embeddings
    embedding1 = embedding1 / norm(embedding1)
    embedding2 = embedding2 / norm(embedding2)

    distance = norm(embedding1 - embedding2)
    print(f"Comparing embeddings: Distance = {distance}, Threshold = {threshold}")  # Debugging info
    return distance < threshold  # Return match status based on distance

