# main.py
import sys
import os
import cv2
import face_recognition
from facial_recognition import get_face_encoding, recognize_face
from database import store_in_database, initialize_database

# Initialize the database
initialize_database()

def add_criminal_to_database(image_path, name):

    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)

    face_encodings = get_face_encoding(image_path)
    if len(face_encodings) == 0:
        # No face detected
        print("No face detected in the image. Please provide a valid image.")
        return
    elif len(face_encodings) > 1:
        # Multiple faces detected
        print(f"Multiple faces detected ({len(face_encodings)} faces). Please provide an image with a single face.")
        return
    else:
        # Only one face detected, proceed to store in the database
        face_encoding = face_encodings[0]
        store_in_database(name, face_encoding)
        print(f"Stored {name}'s face encoding in the database.")
    # if face_encoding is not None:
    #     store_in_database(name, face_encoding)
    #     print(f"Stored {name}'s face encoding in the database.")
    # else:
    #     print("No face detected in the image.")

def detect_criminal_in_image(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    criminals_detected = []

    for face_encoding in face_encodings:
        name = recognize_face(face_encoding)
        if name != "Unknown":
            criminals_detected.append(name)

    if criminals_detected:
        print(f"Criminals detected: {', '.join(criminals_detected)}")
    else:
        print("No criminals detected.")

def detect_criminal_in_video(video_path=None):
    # If a video path is provided, use that; otherwise, use the webcam
    if video_path:
        video_capture = cv2.VideoCapture(video_path)
    else:
        video_capture = cv2.VideoCapture(0)  # Use webcam

    last_criminal = None

    if not video_capture.isOpened():
        print(f"Error: Unable to open video source {video_path}")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = recognize_face(face_encoding)

                if name != "Unknown" and name != last_criminal:
                    print(f"Criminal detected: {name}")
                    last_criminal = name

                if name != "Unknown":
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [<args>]")
        print("Commands:")
        print("  add_criminal <image_path> <name>      Add a criminal to the database")
        print("  detect_image <image_path>             Detect criminal in an image")
        print("  detect_video <video_path>             Detect criminal in a video (optional video_path, defaults to webcam)")
        sys.exit(1)

    command = sys.argv[1]

    if command == "add_criminal":
        if len(sys.argv) != 4:
            print("Usage: python main.py add_criminal <image_path> <name>")
        else:
            image_path = sys.argv[2]
            name = sys.argv[3]
            add_criminal_to_database(image_path, name)

    elif command == "detect_image":
        if len(sys.argv) != 3:
            print("Usage: python main.py detect_image <image_path>")
        else:
            image_path = sys.argv[2]
            detect_criminal_in_image(image_path)

    elif command == "detect_video":
        if len(sys.argv) == 3:
            video_path = sys.argv[2]
            detect_criminal_in_video(video_path)
        else:
            # No video path provided, so use webcam
            detect_criminal_in_video()

    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py <command> [<args>]")
        sys.exit(1)
