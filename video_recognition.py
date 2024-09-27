# video_recognition.py
import cv2
import face_recognition
from facial_recognition import recognize_face

# Real-time video face recognition
def recognize_in_video(video_path=None):
 # Use webcam or video file

    if video_path:
        video_capture = cv2.VideoCapture(video_path)
    else:
        video_capture = cv2.VideoCapture(0)  # Use webcam

    if not video_capture.isOpened():
        print(f"Error: Unable to open video source {video_path}")
        return


    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to grab frame")
            break
        rgb_frame = frame[:, :, ::-1]  # Convert frame from BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_locations:
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = recognize_face(face_encoding)

                if name != "Unknown":  # Only highlight faces that are recognized
                    # Draw rectangle and display the name if face is recognized
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
