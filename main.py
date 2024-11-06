import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image

# Parameters
img_size = 160  # FaceNet input size is 160x160
batch_size = 5

# 1. Load and Preprocess the Images
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Define dataset path
dataset_path = r"C:\Users\Admin\Documents\GitHub\attendance-with-load-control\dataset"  # Use raw string for Windows paths

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='sparse'
)

# 2. Initialize MTCNN and InceptionResnetV1 (FaceNet)
mtcnn = MTCNN(keep_all=True)  # MTCNN for face detection
inception_model = InceptionResnetV1(pretrained='vggface2').eval()  # Pretrained FaceNet model

# 3. Extract Embeddings and Labels
def extract_embeddings(data):
    embeddings = []
    labels = []
    data.reset()  # Reset generator to ensure proper iteration
    for i in range(len(data)):
        img_batch, label_batch = data.next()
        
        # Convert image batch to PIL Image for MTCNN
        pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in img_batch]
        
        # Detect faces and get embeddings
        for i, pil_img in enumerate(pil_images):
            faces, _ = mtcnn.detect(pil_img)  # Detect faces in the image
            if faces is not None:  # If faces are detected
                for face in faces:
                    aligned_face = pil_img.crop((face[0], face[1], face[2], face[3]))  # Crop face from image
                    aligned_face = aligned_face.resize((img_size, img_size))
                    aligned_face = np.array(aligned_face)
                    aligned_face = aligned_face / 255.0  # Normalize the face image
                    aligned_face = torch.tensor(aligned_face).permute(2, 0, 1).unsqueeze(0)  # Convert to tensor

                    # Get embedding using FaceNet model
                    embedding = inception_model(aligned_face).detach().cpu().numpy()
                    embeddings.append(embedding)
                    labels.append(label_batch[i])

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    return embeddings, labels

embeddings, labels = extract_embeddings(train_data)

# 4. Train SVM Classifier
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
clf = SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 6. Recognize New Faces
def recognize_face(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0  # Normalize
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)  # Convert to tensor

    # Get embedding using FaceNet model
    embedding = inception_model(img_tensor).detach().cpu().numpy()
    prediction = clf.predict(embedding)
    return prediction

# Example: Test on a new image
print("Predicted label:", recognize_face("dataset/person1/image1.jpg"))
