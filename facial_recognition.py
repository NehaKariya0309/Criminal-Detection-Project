# face_recognition.py
import numpy as np
import face_recognition
from sklearn.neighbors import NearestNeighbors

from database import get_all_criminals

# Get face encoding for a given image
def get_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    print("face locations:",face_locations)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    print("face encodings:",face_encodings)
    
    if face_encodings:
        return face_encodings
        # return face_encodings[0]  # Return the first detected face encoding
    return None

# Recognize face by comparing with criminal database
def recognize_face(new_face_encoding):
    criminals = get_all_criminals()
    encodings = [np.frombuffer(c[1], dtype=np.float64) for c in criminals]
    names = [c[0] for c in criminals]

    print(f"names : {names}")

    n_neighbors = min(3, len(encodings))


    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(encodings)
    distances, indices = knn.kneighbors([new_face_encoding])

    print(f"Distances: {distances}")
    print(f"Indices: {indices}")

    # Ensure valid indices and distances
    if len(distances) > 0 and len(indices) > 0:
        # Filter based on distance threshold
        closest_names = [names[indices[0][i]] for i in range(len(indices[0])) if distances[0][i] < 0.5]

        # Debug print statement to see the closest names found
        print(f"Closest names: {closest_names}")

        if closest_names:
            return max(set(closest_names), key=closest_names.count)

    return "Unknown"


    
    # for criminal in criminals:
    #     name, encoding_blob = criminal
    #     stored_encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        
    #     distance = np.linalg.norm(stored_encoding - new_face_encoding)
        
    #     if distance < 0.5:  # Threshold for matching
    #         return name
    # return "Unknown"
