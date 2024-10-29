import mysql.connector
import numpy as np

# Connect to the MySQL database
def connect_db():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',    
        password='',  
        database='face_recognition_db'
    )
    return conn

# Insert face embedding into the database
def store_face_embedding(name, embedding):
    conn = connect_db()
    cursor = conn.cursor()

    # Convert numpy array to bytes for storage
    embedding_bytes = embedding.tobytes()

    query = "INSERT INTO faces (name, embedding) VALUES (%s, %s)"
    cursor.execute(query, (name, embedding_bytes))

    conn.commit()
    cursor.close()
    conn.close()

# Retrieve all stored embeddings
def load_face_embeddings():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT name, embedding FROM faces")
    rows = cursor.fetchall()

    # Convert embeddings from BLOB back to numpy array
    face_embeddings = [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

    cursor.close()
    conn.close()

    return face_embeddings

