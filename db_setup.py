import mysql.connector

# Connect to the MySQL database
def connect_db():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password=''
    )
    return conn

# Create the database and tables
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

# Run the setup function
setup_database()
print("The databse has been created successfully! :)")