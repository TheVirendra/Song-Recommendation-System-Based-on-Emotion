import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QStatusBar
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer, Qt
import cv2
import mediapipe as mp
from deepface import DeepFace

class EmotionDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # Load the emotion detection model
        self.model = DeepFace.build_model("Emotion")
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        self.cap = cv2.VideoCapture(0)

        # Setup timer for updating the frame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

     
        self.weights_h5_path = 'modelv1.h5'
        self.weights_keras_path = 'modelv1.keras'

    def init_ui(self):
        # Initialize the UI components
        self.setGeometry(100, 100, 1024, 768)
        self.setWindowTitle('3D Face Emotion Recognition')

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("border: 1px solid black;")

        self.grayface_label = QLabel(self)
        self.grayface_label.setStyleSheet("border: 1px solid black;")

        self.emotion_label = QLabel(self)
        self.emotion_label.setFont(QFont('Arial', 20))
        self.emotion_label.setAlignment(Qt.AlignCenter)
        self.emotion_label.setStyleSheet("font-weight: bold; color: red;")

        self.start_button = QPushButton('Start Live Testing', self)
        self.start_button.setStyleSheet(
            "font-size: 18px; color: white; background-color: #007BFF; padding: 10px 20px; border-radius: 5px;"
        )
        self.start_button.clicked.connect(self.start_live_testing)

        # Add widgets to the layout
        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.grayface_label)
        self.layout.addWidget(self.emotion_label)
        self.layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

    def start_live_testing(self):
        # Start or stop the live testing
        if not self.timer.isActive():
            self.timer.start(30)
            self.start_button.setText('Stop Live Testing')
            self.status_bar.showMessage("Live testing started", 5000)
        else:
            self.timer.stop()
            self.start_button.setText('Start Live Testing')
            self.status_bar.showMessage("Live testing stopped", 5000)

    def update_frame(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray_frame[y:y + h, x:x + w]
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
            normalized_face = resized_face / 255.0
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)

            # Predict emotion
            preds = self.model.predict(reshaped_face)[0]
            emotion_idx = preds.argmax()
            emotion = self.emotion_labels[emotion_idx]

            # Draw rectangle and label on the original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Convert grayscale face ROI to BGR for drawing face mesh
            face_roi_bgr = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2BGR)
            rgb_face_roi = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_face_roi)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        face_roi_bgr, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
                    )

            self.display_grayface(face_roi_bgr)
            self.emotion_label.setText(f"Detected Emotion: {emotion}")

        self.display_frame(frame)

    def display_grayface(self, grayface):
        # Display the grayscale face with face mesh
        resized_grayface = cv2.resize(grayface, (200, 200), interpolation=cv2.INTER_AREA)
        height, width, channel = resized_grayface.shape
        bytes_per_line = 3 * width
        q_img = QImage(resized_grayface.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.grayface_label.setPixmap(QPixmap(q_img))

    def display_frame(self, frame):
        # Display the main video frame
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap(q_img))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionDetectionApp()
    window.show()
    sys.exit(app.exec_())



