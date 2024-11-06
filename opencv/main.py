import sys
import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QImage, QPixmap

# Load known face encodings and names from the dataset
def load_known_faces(dataset_path='dataset'):
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(person_name)

    return known_face_encodings, known_face_names

# Mark attendance in CSV file
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    # Load or create attendance log
    try:
        attendance_df = pd.read_csv('attendance.csv')
    except FileNotFoundError:
        attendance_df = pd.DataFrame(columns=['Name', 'Date', 'Time'])

    # Log only if the person is not already logged in for the day
    if not ((attendance_df['Name'] == name) & (attendance_df['Date'] == date_str)).any():
        attendance_df = attendance_df.append({'Name': name, 'Date': date_str, 'Time': time_str}, ignore_index=True)
        attendance_df.to_csv('attendance.csv', index=False)

# Thread for video capture
class VideoCaptureThread(QThread):
    frame_captured = pyqtSignal(QImage)
    recognized_face = pyqtSignal(str)

    def __init__(self, dataset_path='dataset'):
        super().__init__()
        self.dataset_path = dataset_path
        self.known_face_encodings, self.known_face_names = load_known_faces(dataset_path)
        self.cap = cv2.VideoCapture(0)
        self.is_running = True

    def run(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert the image from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compare face with known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = face_distances.argmin() if matches else None
                if best_match_index is not None and matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    mark_attendance(name)  # Mark attendance when recognized

                    self.recognized_face.emit(name)

                # Draw rectangle around the face
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Convert frame to QImage and emit signal to update UI
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.frame_captured.emit(q_img)

    def stop(self):
        self.is_running = False
        self.quit()

# Main Window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # UI Setup
        self.setWindowTitle("Facial Recognition Attendance System")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        # Camera feed display
        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        # Start button to initiate face recognition
        self.start_button = QPushButton("Start Face Recognition", self)
        self.start_button.clicked.connect(self.start_face_recognition)
        self.layout.addWidget(self.start_button)

        # Attendance table
        self.attendance_table = QTableWidget(self)
        self.attendance_table.setColumnCount(3)
        self.attendance_table.setHorizontalHeaderLabels(['Name', 'Date', 'Time'])
        self.layout.addWidget(self.attendance_table)

        # Main widget and layout
        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

        # Start video capture thread
        self.video_thread = VideoCaptureThread()
        self.video_thread.frame_captured.connect(self.update_frame)
        self.video_thread.recognized_face.connect(self.update_attendance)
        self.video_thread.start()

    def start_face_recognition(self):
        print("Face recognition started...")

    def update_frame(self, q_img):
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def update_attendance(self, name):
        print(f"Attendance marked for: {name}")
        self.load_attendance()

    def load_attendance(self):
        try:
            attendance_df = pd.read_csv('attendance.csv')
        except FileNotFoundError:
            attendance_df = pd.DataFrame(columns=['Name', 'Date', 'Time'])

        self.attendance_table.setRowCount(len(attendance_df))
        for row_idx, row in attendance_df.iterrows():
            self.attendance_table.setItem(row_idx, 0, QTableWidgetItem(row['Name']))
            self.attendance_table.setItem(row_idx, 1, QTableWidgetItem(row['Date']))
            self.attendance_table.setItem(row_idx, 2, QTableWidgetItem(row['Time']))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
