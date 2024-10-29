import cv2
import time
import numpy as np
import database
from face_recognition import detect_faces, process_face, is_match
import os

def draw_text(frame, text, position, color=(0, 255, 255), font_scale=0.7, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def load_face():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error accessing the camera.")
            return

        # Set window to be always on top
        cv2.namedWindow("Face Detection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Face Detection", cv2.WND_PROP_TOPMOST, 1)

        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image from camera.")
                break

            faces = detect_faces(frame)
            elapsed_time = int(time.time() - start_time)
            countdown = 5 - elapsed_time
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                center = (x + w // 2, y + h // 2)
                radius = w // 2
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                draw_text(frame, f"Face detected. Please stay still. Taking photo in {countdown}s...", (10, 30))

                if countdown <= 0:
                    face_embedding = process_face(frame, faces[0])
                    cap.release()
                    cv2.destroyAllWindows()

                    person_name = input("Enter the person's name: ")
                    if person_name:
                        try:
                            database.store_face_embedding(person_name, face_embedding)
                            print(f"Face of {person_name} has been saved to the database.")
                        except Exception as e:
                            print(f"Error saving to database: {str(e)}")
                    else:
                        print("Name input canceled or invalid.")
                    
                    input("Press Enter to return to the menu...")
                    os.system('cls' if os.name == 'nt' else 'clear')
                    break

            draw_text(frame, "Detecting face...", (10, 30))
            draw_text(frame, f"Taking photo in {countdown}s...", (10, 60))
            cv2.imshow("Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or countdown <= 0:
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error processing face: {str(e)}")

def scan_face_with_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to access the camera.")
        return

    # Set window to be always on top
    cv2.namedWindow("Face Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Face Detection", cv2.WND_PROP_TOPMOST, 1)

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break

        faces = detect_faces(frame)
        elapsed_time = int(time.time() - start_time)
        countdown = 5 - elapsed_time
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            center = (x + w // 2, y + h // 2)
            radius = w // 2
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            draw_text(frame, f"Face detected. Please stay still. Taking photo in {countdown}s...", (10, 30))

            if countdown <= 0:
                face_embedding = process_face(frame, faces[0])
                known_faces = database.load_face_embeddings()
                cap.release()
                cv2.destroyAllWindows()

                found_match = False
                for name, known_embedding in known_faces:
                    if is_match(face_embedding, known_embedding):
                        print(f"Voila! You have been found in the db, {name}")
                        found_match = True
                        break

                if not found_match:
                    print("You are not known.")
                
                input("Press Enter to return to the menu...")
                os.system('cls' if os.name == 'nt' else 'clear')
                break

        draw_text(frame, "Detecting face...", (10, 30))
        draw_text(frame, f"Taking photo in {countdown}s...", (10, 60))
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or countdown <= 0:
            break

    cap.release()
    cv2.destroyAllWindows()

def show_menu():
    print("\n" + "=" * 40)
    print("  Welcome to the Face Recognition System!  ")
    print("=" * 40 + "\n")
    print("  1. Scan face with camera to store it  ")
    print("  2. Scan face with camera for verification  ")
    print("  3. Exit  ")
    print("\n" + "=" * 40)
    choice = input("Please enter your choice (1, 2, or 3): ").strip()
    return choice

def main():
    while True:
        choice = show_menu()
        if choice == '1':
            load_face()
        elif choice == '2':
            scan_face_with_camera()
        elif choice == '3':
            print("Exiting the application.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
