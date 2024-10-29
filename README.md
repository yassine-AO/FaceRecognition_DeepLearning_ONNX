# 🧠 Deep Learning Face Recognition with ArcFace and ONNX

## Overview

This project is a robust face recognition application that detects faces, generates embeddings using the ArcFace model, and stores these embeddings in a MySQL database. The application is built in Python and uses OpenCV for face detection, ONNX for efficient model inference, and NumPy for handling array operations. This system is highly suitable for real-time face recognition and verification applications.

---

## Table of Contents

1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [Project Structure](#project-structure)
4. [Detailed Code Explanation](#detailed-code-explanation)
5. [Setup Instructions](#setup-instructions)
6. [How It Works: Face Recognition Pipeline](#how-it-works-face-recognition-pipeline)
7. [Why Use ArcFace and ONNX](#why-use-arcface-and-onnx)
8. [Troubleshooting](#troubleshooting)
9. [Future Enhancements](#future-enhancements)

---

## Features

- **Real-Time Face Detection**: Detect faces in real-time using OpenCV’s Haar Cascade classifier.
- **Face Embedding Generation**: Use ArcFace (an advanced model for face recognition) to create unique 512-dimensional embeddings for each face.
- **Face Verification**: Compares new faces with stored embeddings in the database to determine if a match exists.
- **Database Storage**: Store user face embeddings and details in a MySQL database for efficient face retrieval and verification.

---

## Technologies Used

- **Python**: Core programming language for all operations, including image processing, database interactions, and model inference.
- **OpenCV**: For face detection, capturing, and displaying images.
- **ONNX (Open Neural Network Exchange)**: Model format that allows for efficient inference of the ArcFace model.
- **ArcFace Model**: Deep learning model designed specifically for face recognition and verification tasks.
- **MySQL**: Database for storing user face embeddings.
- **NumPy**: For efficient array and numerical operations, especially during image preprocessing and embedding manipulation.

---

## Project Structure

1. [**app.py**](http://app.py/): The main application file that interacts with the user and provides options to add and verify faces.
2. **face_recognition.py**: Handles face detection, embedding generation, and verification functions.
3. **db_setup.py**: Sets up the MySQL database and the `faces` table.
4. [**database.py**](http://database.py/): Handles database operations like storing and retrieving embeddings.

---

## Detailed Code Explanation

### 1. `app.py` - Main Application Logic

This file serves as the main interface for the application, providing users with options to add a face to the database or verify an existing face. Let’s go over each part of this file.

### Code Snippets and Explanations

```python
import cv2
import database
from face_recognition import detect_faces, process_face, is_match

```

- **Imports**: Imports OpenCV for camera handling, `database` for managing MySQL interactions, and `detect_faces`, `process_face`, and `is_match` from `face_recognition.py` to detect, embed, and compare faces.

### Face Loading Function

```python
def load_face():
    """Capture an image from the camera, detect the face, and store the embedding in the database."""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error accessing the camera.")
            return

```

- **Purpose**: Captures an image from the camera and attempts to detect a face.
- **Explanation**: This function first checks if the camera is accessible; if not, it outputs an error.

```python
        # Capture a single frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            cap.release()
            return

```

- **Purpose**: Captures a single frame, releases the camera if unsuccessful.
- **Explanation**: Captures one frame from the camera, which will be analyzed for face detection.

```python
        faces = detect_faces(frame)
        if len(faces) > 0:
            face_embedding = process_face(frame, faces[0])

```

- **Face Detection and Embedding**: Calls the `detect_faces()` function (from `face_recognition.py`) to locate faces, then processes the detected face to generate its embedding.

---

### 2. `face_recognition.py` - Face Detection and Embedding Generation

This file handles the core functionality: face detection, embedding generation using the ArcFace model, and comparison for verification.

### Code Snippets and Explanations

```python
import cv2
import numpy as np
import onnxruntime as ort
from numpy.linalg import norm

```

- **Imports**: OpenCV for face detection, NumPy for array manipulations, ONNX for ArcFace model inference, and `norm` for calculating the distance between embeddings.

### Loading the ArcFace Model

```python
model_path = 'model/arcface.onnx'
session = ort.InferenceSession(model_path)

```

- **ArcFace Model**: Loads the ArcFace model, which has been pre-trained for face recognition. ArcFace generates 512-dimensional embeddings for faces, providing unique "fingerprints" that make matching effective.
- **ONNX Runtime**: Using ONNX Runtime allows for optimized and efficient inference, crucial for real-time applications.

### Face Detection Function

```python
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

```

- **Face Detection with Haar Cascade**: Converts the image to grayscale (which improves detection accuracy) and detects faces using Haar Cascade.
- **Explanation**: This function returns the coordinates of faces in the image.

### Embedding Generation with ArcFace

```python
def get_face_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    face_pixels = cv2.resize(face_pixels, (112, 112))  # ArcFace input size
    face_pixels = np.expand_dims(face_pixels, axis=0)
    face_pixels = np.transpose(face_pixels, (0, 3, 1, 2))  # Change to NCHW format
    face_pixels = (face_pixels - 127.5) / 128.0  # Normalize

```

- **Explanation**:
    - **NumPy Array**: Converts face pixels to `float32` for compatibility with the model.
    - **Resize and Normalize**: The image is resized to 112x112 pixels (ArcFace requirement) and normalized to ensure consistency.
    - **Format Transformation**: The input is reshaped to match the ArcFace input format (NCHW).
    - **Normalization**: By scaling pixel values, the embedding generation becomes more consistent, crucial for accurate matching.

```python
    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: face_pixels})[0]
    return embedding[0]

```

- **Model Inference**: Feeds the processed image into the ArcFace model to generate the embedding.
- **Purpose**: Returns a unique 512-dimensional embedding representing the face.

### Face Comparison Function

```python
def is_match(embedding1, embedding2, threshold=0.8):
    embedding1 = embedding1 / norm(embedding1)
    embedding2 = embedding2 / norm(embedding2)
    distance = norm(embedding1 - embedding2)
    return distance < threshold

```

- **Explanation**:
    - **Normalization**: Ensures embeddings are comparable.
    - **Distance Calculation**: Uses cosine distance to determine the similarity, with a threshold (0.8) to identify a match.

---

### 3. `db_setup.py` - Database Setup

```python
import mysql.connector

def setup_database():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS face_recognition_db")
    cursor.execute("USE face_recognition_db")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            embedding BLOB
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

```

- **Purpose**: Initializes the database and creates a `faces` table to store face embeddings as BLOBs (Binary Large Objects), ensuring efficient storage and retrieval.

---

### 4. `database.py` - Database Operations

Handles database operations like saving and retrieving face embeddings.

### Code Snippets and Explanations

```python
def store_face_embedding(name, embedding):
    conn = connect_db()
    cursor = conn.cursor()
    embedding_bytes = embedding.tobytes()
    query = "INSERT INTO faces (name, embedding) VALUES (%s, %s)"
    cursor.execute(query, (name, embedding_bytes))
    conn.commit()
    cursor.close()
    conn.close()

```

- **Purpose**: Converts the NumPy embedding to bytes for storage and saves it in the database.

---

## Setup Instructions

### Step 1: Create and Activate a Virtual Environment

1. **Create a Virtual Environment**:
    
    ```bash
    python -m venv venv
    
    ```
    
2. **Activate the Virtual Environment**:
    - **On Windows**:
        
        ```bash
        venv\Scripts\activate
        
        ```
        
    - **On macOS/Linux**:
        
        ```bash
        source venv/bin/activate
        
        ```
        

### Step 2: Install Dependencies

With the virtual environment activated, install the required dependencies:

```bash
pip install -r requirements.txt

```

### Step 3: Download the ArcFace ONNX Model

1. Open the text file named `modelLink` in your project directory.
2. Copy the link provided in the file to download the ArcFace ONNX model.
3. Save the downloaded model file into the `model` folder within your project directory.

### Step 4: Configure the Database

- Update the `db_setup.py` file with your MySQL configuration.
- Just make sure **Mysql** service is **on** , and the database is ready for use
- Run the following command to create the database and table:

```bash
python db_setup.py

```

### Step 5: Run the Application

Finally, start the application with the following command:

```bash
python app.py

```

---

## Why Use ArcFace and ONNX

- **ArcFace**: Known for producing highly discriminative embeddings.
- **ONNX**: Optimized for efficient, real-time inference across platforms.
