import cv2
import os
import numpy as np
import dlib
import joblib

# Initialize the dlib face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor_path = 'opencv/models/shape_predictor_68_face_landmarks.dat'
sp = dlib.shape_predictor(shape_predictor_path)

# Initialize face recognition model
face_rec_model_path = 'opencv/models/dlib_face_recognition_resnet_model_v1.dat'
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

def train_model(dataset_path='dataset', model_path='face_recognizer.pkl', names_path='names.npy'):
    faces = []
    labels = []
    names = {}
    label_counter = 0
    total_images = 0

    # Count total images for progress tracking
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            total_images += len(os.listdir(person_folder))
    
    print(f"Total images to process: {total_images}")

    # Load images and corresponding labels
    image_counter = 0
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image
                faces_rect = face_detector(gray)

                for rect in faces_rect:
                    # Get landmarks and extract face features
                    shape = sp(gray, rect)
                    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)

                    faces.append(np.array(face_descriptor))
                    labels.append(label_counter)

                # Store the name associated with the label
                names[label_counter] = person_name
                label_counter += 1

                # Update progress
                image_counter += 1
                print(f"Processed {image_counter}/{total_images} images", end='\r')

    # Save the names dictionary (it's fine to use np.save for this)
    np.save(names_path, names)

    # Save the trained model using joblib
    with open(model_path, 'wb') as f:
        joblib.dump({'faces': faces, 'labels': labels}, f)

    print(f"\nModel and names saved to {model_path}, {names_path}")
    
    return faces, labels, names

# Train the model
train_model()
