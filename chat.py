import pandas as pd
from flask import Blueprint, request, jsonify, send_from_directory
import random
import os
import re
import google.generativeai as genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

genai.configure(api_key="AIzaSyCMfm3NXxxQEvE_rjEHExk1gW_tYKPnfWw")
google_ai_model = genai.GenerativeModel(model_name="gemini-1.0-pro")

chat_bp = Blueprint('chat', __name__)

df = pd.read_csv(r"songs.csv", encoding="utf-8")
last_emotion = None

analyzer = SentimentIntensityAnalyzer()

emoji_to_emotion = {
    "😊": "happy",
    "😢": "sad",
    "😡": "angry",
    "😌": "calm",
    "❤️": "romantic",
}

def analyze_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    print(f"🔍 Sentiment Analysis Scores: {sentiment_scores}")
    if sentiment_scores['compound'] >= 0.05:
        return "happy"
    elif sentiment_scores['compound'] <= -0.05:
        return "sad"
    else:
        return "neutral"

@chat_bp.route("/send_message", methods=["POST"])
def send_message():
    global last_emotion
    user_message = request.json.get("message", "").lower()
    print(f"📩 Received message: {user_message}")

    detected_emotion = None
    for emoji, emotion in emoji_to_emotion.items():
        if emoji in user_message:
            detected_emotion = emotion
            print(f"🎭 Detected emotion from emoji: {detected_emotion}")
            break

    song_emotions = set()
    for emotion_list in df["emotion"]:
        song_emotions.update([e.strip().lower() for e in emotion_list.split(",")])

    sorted_emotions = sorted(song_emotions, key=len, reverse=True)
    matched_emotion = next((emotion for emotion in sorted_emotions if re.search(rf"\b{re.escape(emotion)}\b", user_message)), None)

    if matched_emotion:
        last_emotion = matched_emotion
        print(f"🔹 Matched emotion from dataset: {matched_emotion}")
    elif detected_emotion:
        last_emotion = detected_emotion
    else:
        last_emotion = analyze_sentiment(user_message)
        print(f"📊 Sentiment-based emotion: {last_emotion}")

    return recommend_songs()

@chat_bp.route("/recommend_songs", methods=["GET"])
def recommend_songs():
    global last_emotion
    response = f"🎵 Here's a song for your mood ({last_emotion}): "

    filtered_songs = df[df["emotion"].str.contains(rf"\b{re.escape(last_emotion)}\b", case=False, na=False)][["song_title", "file_path"]].to_dict(orient="records")
    print(f"🎼 Filtered songs: {filtered_songs}")

    if filtered_songs:
        song = random.choice(filtered_songs)
        response += f" {song['song_title']}"
        song_path = f"/chat/audio/{song['file_path']}"
    else:
        response += " I couldn't find a song for that mood. Try another emotion!"
        song_path = ""

    return jsonify({"response": response, "path": song_path, "song_title": song['song_title'] if filtered_songs else "No song found"})

SONG_FOLDER = "audio"

@chat_bp.route("/get_songs_by_singer", methods=["POST"])
def get_songs_by_singer():
    data = request.get_json()
    singer = data.get("singer")
    singer_folder = os.path.join(SONG_FOLDER, singer)

    if os.path.exists(singer_folder):
        songs = [f for f in os.listdir(singer_folder) if f.endswith(".mp3")]
        return jsonify({"songs": songs, "folder": f"/{singer_folder}"})
    else:
        return jsonify({"songs": []}), 404

@chat_bp.route("/get_next_song", methods=["POST"])
def get_next_song():
    global last_emotion
    if last_emotion:
        filtered_songs = df[df["emotion"].str.contains(rf"\b{re.escape(last_emotion)}\b", case=False, na=False)][["song_title", "file_path"]].to_dict(orient="records")
        if filtered_songs:
            song = random.choice(filtered_songs)
            song_path = f"/chat/audio/{song['file_path']}"
            return jsonify({"path": song_path, "song_title": song['song_title']})
    return jsonify({"path": "", "song_title": "No song found"})

@chat_bp.route('/audio/<path:filename>')
def send_music(filename):
    return send_from_directory(os.path.join('audio'), filename)

@chat_bp.route("/get_recommendation", methods=["POST"])
def get_recommendation():
    global last_emotion
    user_message = request.json.get("message", "").lower()
    print(f"Received message: {user_message}")

    predefined_responses = {
        "hi": "Hello! How are you today? 😊",
        "hello": "Hello! How are you today? 😊",
        "how are you": "I'm an AI, but I'm here to help! 🎵",
        "thanks": "You're welcome! Enjoy your music! 🎶",
        "what is your name": "I'm Chitti, your AI music buddy! 🤖",
        "who are you": "I'm Chitti, your personal song recommender AI! 🎶"
    }
    
    for key, value in predefined_responses.items():
        if key in user_message:
            print(f"Matched predefined response: {key} -> {value}")
            return jsonify({"response": value})

    song_emotions = set()
    for emotion_list in df["emotion"]:
        song_emotions.update([e.strip().lower() for e in emotion_list.split(",")])

    sorted_emotions = sorted(song_emotions, key=len, reverse=True)
    matched_emotion = next((emotion for emotion in sorted_emotions if re.search(rf"\b{re.escape(emotion)}\b", user_message)), None)

    if matched_emotion:
        last_emotion = matched_emotion
        response = f"Here's a song for {matched_emotion}: "
    else:
        detected_emotion = analyze_sentiment(user_message)
        last_emotion = detected_emotion
        response = f"I detected you're feeling {detected_emotion}. Here's a song for you: "

    

    filtered_songs = df[df["emotion"].str.contains(rf"\b{re.escape(last_emotion)}\b", case=False, na=False)][["song_title", "file_path"]].to_dict(orient="records")
    if filtered_songs:
        song = random.choice(filtered_songs)
        response += f" {song['song_title']}"
        song_path = f"/chat/audio/{song['file_path']}"
    else:
        response += " I couldn't find a song for that mood. Try another emotion!"
        song_path = ""

    return jsonify({"response": response, "path": song_path, "song_title": song['song_title'] if filtered_songs else "No song found"})