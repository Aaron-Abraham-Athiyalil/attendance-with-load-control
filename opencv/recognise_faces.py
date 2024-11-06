import cv2
import numpy as np
import dlib
import joblib
import csv
from datetime import datetime
import os

# Initialize the dlib face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor_path = 'opencv/models/shape_predictor_68_face_landmarks.dat'
sp = dlib.shape_predictor(shape_predictor_path)

# Initialize face recognition model
face_rec_model_path = 'opencv/models/dlib_face_recognition_resnet_model_v1.dat'
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# CSV file path for attendance
attendance_file = 'attendance.csv'

def load_trained_model(model_path='face_recognizer.pkl', names_path='names.npy'):
    # Load the trained face recognizer model
    with open(model_path, 'rb') as f:
        model_data = joblib.load(f)

    # Load the names associated with the labels
    names = np.load(names_path, allow_pickle=True).item()

    return model_data, names

def mark_attendance(name):
    # Get the current date and time
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Check if the CSV file exists and write header if it's the first entry
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Timestamp'])  # Write header

    # Append the name and timestamp to the attendance CSV file
    with open(attendance_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, timestamp])
    
    print(f"Attendance marked for: {name} at {timestamp}")

def recognize_face_from_webcam(model_data, names):
    faces = model_data['faces']
    labels = model_data['labels']

    # Set to keep track of names whose attendance has already been marked
    marked_names = set()

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    print("Starting webcam...")

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces_rect = face_detector(gray)

        for rect in faces_rect:
            # Get landmarks and extract face features
            shape = sp(gray, rect)
            face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)

            # Compare the captured face with the stored faces
            distances = np.linalg.norm(faces - np.array(face_descriptor), axis=1)
            min_distance_index = np.argmin(distances)

            # Check if the face is recognized or unknown
            if distances[min_distance_index] < 0.6:  # Threshold to determine a match
                name = names[labels[min_distance_index]]
                
                # Mark attendance only once per person
                if name not in marked_names:
                    mark_attendance(name)  # Mark attendance for the recognized person
                    marked_names.add(name)  # Add the name to the marked set
            else:
                name = "Unknown"

            # Draw rectangle around the face and display the name
            (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame with the recognized faces
        cv2.imshow("Face Recognition", frame)

        # Exit the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Load the trained model and names
    model_data, names = load_trained_model()

    # Start recognizing faces from the webcam
    recognize_face_from_webcam(model_data, names)
