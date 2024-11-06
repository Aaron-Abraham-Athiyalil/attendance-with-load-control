import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionResNetV2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
from facenet_pytorch import MTCNN
import torch

# Parameters
img_size = 160  # Size for the image input to the model
batch_size = 5
dataset_path = "C:/Users/Admin/Documents/GitHub/attendance-with-load-control/dataset"  # Your dataset path

# 1. Load and Preprocess the Images
datagen = ImageDataGenerator(rescale=1.0/255)

# Set up the data generator
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='sparse'
)

# 2. Load the Pre-trained Model (Inception ResNet V2)
def load_inception_resnet_v2():
    base_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    embedding_model = Model(inputs=base_model.input, outputs=x)
    return embedding_model

embedding_model = load_inception_resnet_v2()

# 3. Load MTCNN for Face Detection
mtcnn = MTCNN(keep_all=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 4. Function to Extract Embeddings from Images
def extract_embeddings(data):
    embeddings = []
    labels = []
    
    data.reset()  # Reset data generator to start from the beginning
    print("Starting embedding extraction...")
    
    for i, (img_batch, label_batch) in enumerate(data):
        print(f"Processing batch {i + 1}/{len(data)}")

        # Convert image batch to PIL Image for MTCNN detection
        pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in img_batch]

        # Detect faces and get embeddings
        for j, pil_img in enumerate(pil_images):
            faces, _ = mtcnn.detect(pil_img)  # Detect faces in the image
            if faces is not None:  # If faces are detected
                print(f"Detected {len(faces)} faces in image {i*batch_size + j + 1}")
                for face in faces:
                    aligned_face = pil_img.crop((face[0], face[1], face[2], face[3]))  # Crop face
                    aligned_face = aligned_face.resize((img_size, img_size))  # Resize for the model
                    aligned_face = np.array(aligned_face) / 255.0  # Normalize image
                    aligned_face = np.expand_dims(aligned_face, axis=0)  # Add batch dimension
                    
                    # Extract embedding using Inception ResNet V2 model
                    embedding = embedding_model.predict(aligned_face)
                    embeddings.append(embedding)
                    labels.append(label_batch[j])

        # Stop if we've processed all the images
        if len(embeddings) >= len(data.filenames):
            break
    
    embeddings = np.vstack(embeddings)  # Combine all embeddings into a single array
    labels = np.array(labels)  # Convert labels to an array
    print(f"Extracted {embeddings.shape[0]} embeddings.")
    return embeddings, labels

# Extract embeddings from training data
embeddings, labels = extract_embeddings(train_data)

# 5. Train the SVM Classifier
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
clf = SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train)

# 6. Evaluate the Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 7. Recognize New Faces
def recognize_face(image_path):
    img = Image.open(image_path).convert('RGB')
    faces, _ = mtcnn.detect(img)
    
    if faces is None:
        print("No faces detected.")
        return None

    # Assume the first face detected is the target face
    face = faces[0]
    aligned_face = img.crop((face[0], face[1], face[2], face[3]))
    aligned_face = aligned_face.resize((img_size, img_size))  # Resize face
    aligned_face = np.array(aligned_face) / 255.0  # Normalize face
    aligned_face = np.expand_dims(aligned_face, axis=0)  # Add batch dimension

    # Get embedding for the face
    embedding = embedding_model.predict(aligned_face)
    
    # Predict the label using the trained SVM classifier
    prediction = clf.predict(embedding)
    return prediction

# Example: Test on a new image
print("Predicted label:", recognize_face("dataset/person1/image1.jpg"))
