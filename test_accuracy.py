import cv2
from deepface import DeepFace

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]  # Extract face region

        try:
            # Analyze the detected face for emotion
            result = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)
            detected_emotion = result[0]['dominant_emotion']

            # Draw a rectangle around the face and label with the detected emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error detecting emotion: {e}")

    # Display the live video feed with emotion labels
    cv2.imshow("Live Emotion Detection", frame)

    # ✅ **Fix: Properly Detect 'q' Key Press**
    key = cv2.waitKey(1) & 0xFF  # Ensure it reads keypress correctly
    if key == ord('q'):  # Press 'q' to quit
        print("Exiting program...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
