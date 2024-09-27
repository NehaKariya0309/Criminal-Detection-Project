# face_recognition.py
import numpy as np
import face_recognition
from database import get_all_criminals

# Get face encoding for a given image
def get_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    if face_encodings:
        return face_encodings[0]  # Return the first detected face encoding
    return None

# Recognize face by comparing with criminal database
def recognize_face(new_face_encoding):
    criminals = get_all_criminals()
    
    for criminal in criminals:
        name, encoding_blob = criminal
        stored_encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        
        distance = np.linalg.norm(stored_encoding - new_face_encoding)
        
        if distance < 0.5:  # Threshold for matching
            return name
    return "Unknown"
