from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import re
from ml_model import (
    extract_features,
    load_model,
    spotify_recommendations,
    record_audio,
    HAS_SOUNDDEVICE
)

# ------------------- Initialize Flask -------------------
app = Flask(__name__)
app.secret_key = "super_secret_key_123"

# ------------------- Load Trained Model -------------------
MODEL_PATH = "saved_model.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("saved_model.joblib not found! Train model first.")

# Fixed unpacking: ignore the extra 'use_cnn' value
clf, scaler, le, _ = load_model(MODEL_PATH)
if clf is None or scaler is None or le is None:
    raise RuntimeError("Failed to load ML model. Train first.")

print("🎯 ML Model Loaded Successfully!")

# ------------------- NLP Parser -------------------
EMOTIONS = ["happy", "sad", "angry", "calm", "neutral", "fear", "disgust", "surprise"]
LANGUAGES = ["english", "hindi", "marathi"]

def parse_user_input(text):
    """
    Detect emotion and language from user text.
    Defaults: emotion='happy', language='english'
    """
    text = text.lower()
    emotion_found = next((e for e in EMOTIONS if re.search(r"\b" + e + r"\b", text)), "happy")
    language_found = next((l for l in LANGUAGES if re.search(r"\b" + l + r"\b", text)), "english")
    return emotion_found, language_found

# ------------------- Routes -------------------

@app.route("/")
def root():
    return redirect(url_for("index"))

@app.route("/login", methods=["GET"])
def login():
    return render_template("login.html")

@app.route("/index")
@app.route("/index.html")
def index():
    return render_template("index.html", user=None, has_sounddevice=HAS_SOUNDDEVICE)

# ------------------- Audio-based Emotion Detection -------------------
@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    if not HAS_SOUNDDEVICE:
        return jsonify({"error": "Audio recording not supported on this device"})

    try:
        language = request.form.get("language", "english").lower()

        # Record audio
        audio_file = record_audio(seconds=3)
        if not audio_file or not os.path.exists(audio_file):
            return jsonify({"error": "Failed to record audio"})

        # Extract features and predict emotion
        features = extract_features(audio_file)
        scaled_features = scaler.transform([features])
        predicted_label = clf.predict(scaled_features)[0]
        emotion = le.inverse_transform([predicted_label])[0]

        # Spotify recommendations
        songs = spotify_recommendations(emotion, language=language, limit=5)

        return jsonify({
            "success": True,
            "input_type": "audio",
            "emotion": emotion,
            "language": language.capitalize(),
            "songs": songs
        })

    except Exception as e:
        print("Error in /detect_emotion:", e)
        return jsonify({"error": str(e)})

# ------------------- Text-based Song Request -------------------
@app.route("/get_songs", methods=["POST"])
def get_songs():
    try:
        user_text = request.form.get("user_text", "")
        if not user_text:
            return jsonify({"error": "No text input provided"})

        # Parse emotion and language
        emotion, language = parse_user_input(user_text)
        songs = spotify_recommendations(emotion, language=language, limit=5)

        return jsonify({
            "success": True,
            "input_type": "text",
            "emotion": emotion,
            "language": language.capitalize(),
            "songs": songs
        })

    except Exception as e:
        print("Error in /get_songs:", e)
        return jsonify({"error": str(e)})

# ------------------- Run Server -------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
