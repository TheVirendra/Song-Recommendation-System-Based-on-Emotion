# Import required libraries
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session, send_from_directory, Response
import sqlite3
import cv2
import mediapipe as mp
from deepface import DeepFace
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import google.generativeai as genai
import json
import random
import numpy as np
from chat import chat_bp

import os
from datetime import datetime
import nltk

# Flask app setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.register_blueprint(chat_bp, url_prefix='/chat')



# Initialize DeepFace model for emotion detection
Model = DeepFace.build_model("Emotion")
class trainedmodel:
    def dummy_method(self):
        self.weights_h5_path = 'modelv1.h5'
        self.weights_keras_path = 'modelv1.keras'

# Create an instance if needed, but the method isn't called so nothing will execute
model_instance = trainedmodel()


emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

if "RENDER" in os.environ:  # Check if running on Render
    use_camera = False
else:
    use_camera = True

if use_camera:
    cap = cv2.VideoCapture(0)  # This will only run locally
if not cap.isOpened():
    print("Error: Unable to access the camera")
    exit()
# Map emotions to sentiments
def map_emotion(emotion):
    emotion_mapping = {
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "sad": "sad",
        "happy": "happy",
        "surprise": "suprise",
        "neutral": "neutral"
    }
    return emotion_mapping.get(emotion, "neutral")

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

init_db()

# Configure Google Generative AI
genai.configure(api_key="AIzaSyCMfm3NXxxQEvE_rjEHExk1gW_tYKPnfWw")  # Replace with a secure method for production
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}
google_ai_model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    generation_config=generation_config,
)


init_db()

# Set your Google Generative AI API key (ensure this is stored securely, not hardcoded in production)
genai.configure(api_key="AIzaSyCMfm3NXxxQEvE_rjEHExk1gW_tYKPnfWw")  # Replace with a secure method for production

# Define generation configuration
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

# Initialize the chat model session
model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    generation_config=generation_config,
)

SONG_FOLDER = "audio"  # Ensure this matches your actual folder

@app.route("/get_songs_by_singer", methods=["POST"])
def get_songs_by_singer():
    data = request.get_json()
    singer = data.get("singer")

    if not singer:
        return jsonify({"error": "Singer not provided"}), 400

    singer_folder = os.path.join(SONG_FOLDER, singer)

    if os.path.exists(singer_folder):
        songs = [f for f in os.listdir(singer_folder) if f.endswith(".mp3")]
        return jsonify({"songs": songs, "folder": f"/{singer_folder}"})
    else:
        return jsonify({"error": "Singer folder not found"}), 404  # More specific error


@app.route('/')
def home():
      return render_template('home.html')
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Save user data to SQLite
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            conn.close()
            flash('Registration successful! You can now log in.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose a different username.')
            return redirect(url_for('register'))
    
    return render_template('register.html')
current_emotion = "neutral"
def generate_frames():
    global current_emotion
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
            normalized_face = resized_face / 255.0
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)

            # Predict emotion
            preds = Model.predict(reshaped_face)[0]
            emotion_idx = preds.argmax()
            detected_emotion = emotion_labels[emotion_idx]

            # Map the detected emotion to its "opposite"
            emotion = map_emotion(detected_emotion)

            # Update the global emotion variable
            current_emotion = emotion

            # Draw rectangle and label on the original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/audio/<path:filename>')
def send_audio(filename):
    return send_from_directory('audio', filename)    

@app.route('/current_emotion')
def current_emotion_endpoint():
    global current_emotion
    return {'emotion': current_emotion}

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Retrieve user data from SQLite
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            session['username'] = username
            session.pop('chat_history', None)  # Clear chat history on new login
            session['chat_count'] = 0  # Initialize chat count
            return redirect(url_for('homes'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    
    return render_template('login.html')
@app.route('/camera1')
def camera1():
    if 'username' in session:
        return render_template('camera1.html')  # Ensure you create this template
    else:
        flash('Please log in first.')
        return redirect(url_for('login'))

@app.route('/homes')
def homes():
    if 'username' in session:
        return render_template('homes.html', username=session['username'])
    else:
        flash('Please log in first.')
        return redirect(url_for('login')) 

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('chat_history', None)  # Clear chat history on logout
    session.pop('chat_count', None)  # Clear chat count on logout
    flash('You have been logged out.')
    return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html')

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

@app.route('/send_message', methods=['POST'])
def send_message():
    user_input = request.form['user_input']
    show_recommendations = request.form.get('show_button', 'false') == 'true'

    if not user_input.strip():
        flash('Message cannot be empty.')
        return redirect(url_for('chat'))

    if 'chat_history' not in session:
        session['chat_history'] = []

    first_interaction = session.get('first_interaction', True)

    if first_interaction:
        session['first_interaction'] = False
        session['chat_count'] = 1
        ai_response = "Hello! I'm your educational assistant. How are you today?"
        session['chat_history'].append({'role': 'assistant', 'message': ai_response})
        return render_template('chat1.html', chat_history=session['chat_history'], show_button=False)

    session['chat_count'] = session.get('chat_count', 0) + 1

    # Sentiment Analysis
    sentiment_scores = sid.polarity_scores(user_input)
    sentiment_label = "positive" if sentiment_scores['compound'] >= 0.05 else "negative" if sentiment_scores['compound'] <= -0.05 else "neutral"
    session['chat_history'].append({'role': 'user', 'message': user_input, 'sentiment': sentiment_label})

    positive_count = sum(1 for msg in session['chat_history'] if msg.get('role') == 'user' and msg.get('sentiment') == 'positive')
    negative_count = sum(1 for msg in session['chat_history'] if msg.get('role') == 'user' and msg.get('sentiment') == 'negative')
    neutral_count = sum(1 for msg in session['chat_history'] if msg.get('role') == 'user' and msg.get('sentiment') == 'neutral')

    if positive_count > negative_count and positive_count > neutral_count:
        overall_sentiment = "positive"
    elif negative_count > positive_count and negative_count > neutral_count:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    session['overall_sentiment'] = overall_sentiment

    if len(session['chat_history']) > 8:
        session['chat_history'] = session['chat_history'][-8:]

    # AI response generation
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(user_input)

    try:
        if response.candidates and len(response.candidates) > 0:
            ai_response = response.candidates[0].content.parts[0].text
        else:
            ai_response = "No valid response received."
    except (IndexError, AttributeError) as e:
        print("Error accessing response content:", e)
        ai_response = "Sorry, I couldn't understand that."

    session['chat_history'].append({'role': 'assistant', 'message': ai_response})

    # Display the button after 5 interactions
    show_button = session['chat_count'] >= 3
    button_label = "Feeling Happy or Good" if overall_sentiment == "positive" else "sad or negative" if overall_sentiment == "negative" else "Neutral"

    # Redirect to recommendations page if show button was clicked
    if show_recommendations:
        return redirect(url_for('recommend_songs'))

    return render_template(
        'chat1.html',
        chat_history=session['chat_history'],
        overall_sentiment=overall_sentiment,
        show_button=show_button,
        button_label=button_label
    )


@app.route('/recommend_songs')
def recommend_songs():
    if 'overall_sentiment' not in session and 'emotion' not in request.args:
        flash("No sentiment data found. Please interact with the system first.")
        return redirect(url_for('chat'))

    # Determine the sentiment/emotion
    emotion = request.args.get('emotion', session.get('overall_sentiment', None))
    if not emotion:
        flash("No emotion data available. Please try again.")
        return redirect(url_for('chat'))

    # Load recommended songs based on sentiment
    with open('90s_bollywood_songs.json', 'r', encoding='utf-8') as file:
        songs_data = json.load(file)
    

    # Select 7 random songs for the given emotion
    if emotion not in songs_data:
        flash(f"No songs available for the emotion: {emotion}")
        return redirect(url_for('chat'))

    recommended_songs = random.sample(songs_data.get(emotion, []), min(len(songs_data.get(emotion, [])), 19))

    return render_template('recommendations.html', recommended_songs=recommended_songs, sentiment=emotion)




@app.route('/chat')
def chat():
    if 'username' in session:
        return render_template('chat1.html')  # Replace with your actual chat template
    else:
        flash('Please log in first.')
        return redirect(url_for('login'))

# Start the application
if __name__ == "__main__":
    port = os.getenv("PORT")  # Get PORT from environment variable
    if port is None or not port.isdigit():
        port = 5000  # Default to port 5000 if PORT is invalid
    app.run(host="0.0.0.0", port=int(port))
