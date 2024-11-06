import sys
import cv2
import numpy as np
import dlib
import joblib
import csv
from datetime import datetime
import os
from PyQt5.QtWidgets import QPushButton

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QStackedWidget, QFrame
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# Initialize the dlib face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor_path = 'opencv/models/shape_predictor_68_face_landmarks.dat'
sp = dlib.shape_predictor(shape_predictor_path)

# Initialize face recognition model
face_rec_model_path = 'opencv/models/dlib_face_recognition_resnet_model_v1.dat'
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# CSV file path for attendance
attendance_file = 'attendance.csv'

# Load the trained model and names
def load_trained_model(model_path='face_recognizer.pkl', names_path='names.npy'):
    with open(model_path, 'rb') as f:
        model_data = joblib.load(f)

    names = np.load(names_path, allow_pickle=True).item()
    return model_data, names

# Mark attendance to the CSV file
def mark_attendance(name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Timestamp'])

    with open(attendance_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, timestamp])

    print(f"Attendance marked for: {name} at {timestamp}")

class FaceRecognitionThread(QThread):
    update_frame_signal = pyqtSignal(QImage)
    mark_attendance_signal = pyqtSignal(str)

    def __init__(self, model_data, names):
        super().__init__()
        self.model_data = model_data
        self.names = names
        self.cap = cv2.VideoCapture(0)
        self.faces = self.model_data['faces']
        self.labels = self.model_data['labels']
        self.marked_names = set()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces_rect = face_detector(gray)

            for rect in faces_rect:
                shape = sp(gray, rect)
                face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)

                # Compare captured face with stored faces
                distances = np.linalg.norm(self.faces - np.array(face_descriptor), axis=1)
                min_distance_index = np.argmin(distances)

                if distances[min_distance_index] < 0.6:  # Threshold for face match
                    name = self.names[self.labels[min_distance_index]]

                    if name not in self.marked_names:
                        self.mark_attendance_signal.emit(name)
                        self.marked_names.add(name)
                else:
                    name = "Unknown"

                (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Convert frame to QImage for PyQt display
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()

            self.update_frame_signal.emit(q_img)

    def stop(self):
        self.cap.release()
        self.quit()

class HomePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel("Welcome to the Face Recognition System!")
        self.label.setStyleSheet("color: black; font-size: 24px;")
        layout.addWidget(self.label)
        self.setLayout(layout)

class FaceRecognitionPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Attendance")

        # Layout setup
        layout = QVBoxLayout()

        # Video label to display webcam feed
        self.video_label = QLabel(self)
        layout.addWidget(self.video_label)

        # Attendance message
        self.attendance_label = QLabel("No attendance marked yet.", self)
        layout.addWidget(self.attendance_label)

        # Relay checkbox layout
        self.relay_checkboxes_layout = QVBoxLayout()
        self.relay_checkboxes = []
        for i in range(5):  # Now there are 5 relays
            relay_checkbox = QCheckBox(f"Relay {i+1} - OFF", self)
            relay_checkbox.setStyleSheet("color: white; font-size: 18px;")
            relay_checkbox.toggled.connect(self.toggle_relay)
            self.relay_checkboxes.append(relay_checkbox)
            self.relay_checkboxes_layout.addWidget(relay_checkbox)

        layout.addLayout(self.relay_checkboxes_layout)
        self.setLayout(layout)

        # Load model data and names
        self.model_data, self.names = load_trained_model()

        # Mapping people to relays (1-based index)
        self.person_to_relay = {
            "Rajat": 0,
            "Rohith": 1,
            "Tejas": 2,
            "Varun": 3,
            "Sudharsshan": 4
        }

        self.face_recognition_thread = None

    def showEvent(self, event):
        # Start the face recognition automatically when the page is shown
        self.start_face_recognition()

    def start_face_recognition(self):
        # Start the face recognition thread
        self.face_recognition_thread = FaceRecognitionThread(self.model_data, self.names)
        self.face_recognition_thread.update_frame_signal.connect(self.update_video_frame)
        self.face_recognition_thread.mark_attendance_signal.connect(self.mark_attendance)
        self.face_recognition_thread.start()

    def update_video_frame(self, q_img):
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def mark_attendance(self, name):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if not os.path.exists(attendance_file):
            with open(attendance_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Name', 'Timestamp'])

        with open(attendance_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, timestamp])

        self.attendance_label.setText(f"Attendance marked for: {name} at {timestamp}")
        print(f"Attendance marked for: {name} at {timestamp}")

        # Turn on the corresponding relay based on the person
        if name in self.person_to_relay:
            relay_index = self.person_to_relay[name]
            self.relay_checkboxes[relay_index].setChecked(True)

    def toggle_relay(self):
        sender = self.sender()
        relay_number = self.relay_checkboxes.index(sender) + 1
        status = "ON" if sender.isChecked() else "OFF"
        sender.setText(f"Relay {relay_number} - {status}")
        print(f"Relay {relay_number} toggled {status}")

    def closeEvent(self, event):
        if self.face_recognition_thread:
            self.face_recognition_thread.stop()
            self.face_recognition_thread.wait()
        event.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Recognition System")
        self.setGeometry(0, 0, 1920, 1080)  # Set to fullscreen
        self.setWindowFlag(Qt.FramelessWindowHint)  # Remove window border

        # Apply a dark theme with dark grey background
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2C2C2C;  # Dark grey background for the main window
            }
            QWidget {
                background-color: #333333;  # Dark grey background for widgets
            }
            QCheckBox {
                color: white;
                font-size: 18px;
            }
            QLabel {
                color: white;
                font-size: 18px;
            }
        """)

        # Set up the stacked widget for navigation
        self.stacked_widget = QStackedWidget(self)
        self.home_page = HomePage()
        self.face_recognition_page = FaceRecognitionPage()

        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.face_recognition_page)

        # Sidebar layout setup
        self.sidebar = QFrame(self)
        sidebar_layout = QVBoxLayout(self.sidebar)

        self.face_recognition_button = QPushButton("Start Face Recognition", self)
        self.face_recognition_button.clicked.connect(self.show_face_recognition_page)
        sidebar_layout.addWidget(self.face_recognition_button)

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close)
        sidebar_layout.addWidget(self.exit_button)

        # Sidebar styling
        self.sidebar.setStyleSheet("""
            QFrame {
                background-color: #444444;
            }
            QPushButton {
                background-color: #444444;
                color: white;
                font-size: 20px;
                border: none;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
        """)

        # Main layout setup
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.stacked_widget)

        container = QWidget(self)
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def show_home_page(self):
        self.stacked_widget.setCurrentWidget(self.home_page)

    def show_face_recognition_page(self):
        self.stacked_widget.setCurrentWidget(self.face_recognition_page)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
