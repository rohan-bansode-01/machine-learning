import os
import re
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import noisereduce as nr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# --- Audio recording support ---
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

# --- Spotify setup ---
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

SPOTIFY_CLIENT_ID = "424dbd9d899c418ea73f685394e26822"
SPOTIFY_CLIENT_SECRET = "dba6bedde1ab4784b566efb849411c22"

try:
    sp = Spotify(auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    ))
    HAS_SPOTIFY = True
    print("✅ Spotify API connected")
except:
    HAS_SPOTIFY = False
    print("⚠️ Spotify failed to connect, using offline mode")

# --- Emotion mapping ---
emotion_map = {
    "01": "sad", "02": "calm", "03": "happy", "04": "neutral",
    "05": "angry", "06": "fear", "07": "disgust", "08": "surprise"
}
reverse_emotion_map = {v: k for k, v in emotion_map.items()}

# -------------------------------------------------------
# AUDIO PREPROCESSING: Noise Reduction + VAD + Normalization
# -------------------------------------------------------
def preprocess_audio(y, sr=22050):
    # Noise reduction
    y = nr.reduce_noise(y=y, sr=sr)

    # Silence removal (VAD)
    intervals = librosa.effects.split(y, top_db=25)
    y_nonsilent = np.concatenate([y[start:end] for start, end in intervals])

    # Peak normalization
    peak = np.max(np.abs(y_nonsilent))
    if peak > 0:
        y_nonsilent = y_nonsilent / peak

    return y_nonsilent

# -------------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------------
def extract_features(path, sr=22050, use_cnn=False):
    y, sr = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y)
    y = preprocess_audio(y, sr=sr)  # Apply noise reduction, VAD, normalization

    if not use_cnn:
        # Traditional features (MFCC + Chroma + Mel + Spec + Tonnetz)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        mel_mean = np.mean(mel, axis=1)

        spec = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_mean = np.mean(spec, axis=1)

        try:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
        except:
            tonnetz_mean = np.zeros(6)

        features = np.hstack([mfcc_mean, mfcc_std, chroma_mean, mel_mean, spec_mean, tonnetz_mean])
        return np.array(features)
    else:
        # CNN features: log-mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel_spec = librosa.power_to_db(mel_spec)
        import cv2
        log_mel_resized = cv2.resize(log_mel_spec, (128, 128))
        log_mel_resized = (log_mel_resized - log_mel_resized.mean()) / (log_mel_resized.std() + 1e-9)
        return log_mel_resized

# -------------------------------------------------------
# RECORD AUDIO
# -------------------------------------------------------
def record_audio(seconds=3, sr=22050, out_path='voice_input.wav'):
    if not HAS_SOUNDDEVICE:
        print("⚠️ sounddevice not available")
        return None
    print("🎙 Recording...")
    recording = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
    sd.wait()
    sf.write(out_path, recording.flatten(), sr)
    print("✅ Saved:", out_path)
    return out_path

# -------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------
def train_from_csv(csv_path="metadata.csv", model_path="saved_model.joblib", use_cnn=False):
    if not os.path.exists(csv_path):
        print("❌ CSV not found")
        return

    df = pd.read_csv(csv_path)
    X, y = [], []

    print("📊 Extracting features...")
    for _, row in df.iterrows():
        fp = row['path']
        label = row['classname']
        if not os.path.exists(fp):
            continue
        try:
            feat = extract_features(fp, use_cnn=use_cnn)
            X.append(feat)
            y.append(label)
        except:
            continue

    X = np.array(X)
    y = np.array(y)

    if not use_cnn:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.reshape(X.shape[0], -1)
        scaler = None

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
    clf.fit(X_scaled, y_enc)

    joblib.dump({'clf': clf, 'scaler': scaler, 'label_encoder': le, 'use_cnn': use_cnn}, model_path)
    print("✅ Model trained and saved:", model_path)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
def load_model(model_path='saved_model.joblib'):
    if not os.path.exists(model_path):
        print("❌ Model not found. Train first.")
        return None, None, None, None
    data = joblib.load(model_path)
    return data['clf'], data['scaler'], data['label_encoder'], data.get('use_cnn', False)

# -------------------------------------------------------
# SPOTIFY RECOMMENDER
# -------------------------------------------------------
def spotify_recommendations(emotion, language='english', limit=5):
    if not HAS_SPOTIFY:
        return [f"Sample {emotion} {language} song"] * limit

    queries = {
    'happy': [
        "happy upbeat pop",
        "feel good dance hits",
        "uplifting cheerful mood",
        "positive vibes energy",
        "summer happy music"
    ],
    
    'sad': [
        "sad emotional songs",
        "heartbreak chill vibes",
        "soft mellow acoustic",
        "lonely piano sad",
        "slow emotional pop"
    ],
    
    'angry': [
        "angry energetic rock",
        "rage metal heavy",
        "intense aggressive beats",
        "high adrenaline workout",
        "hard rock angry mood"
    ],
    
    'neutral': [
        "calm chill background",
        "lofi study beats",
        "soft ambient neutral",
        "focus playlist no vocals",
        "simple relaxing instrumentals"
    ],

    'calm': [
        "relaxing calm peaceful",
        "soothing ambient music",
        "soft acoustic chill",
        "meditation gentle sounds",
        "stress relief calm tunes"
    ],

    'fear': [
        "scary thriller soundtrack",
        "dark horror ambience",
        "suspense cinematic score",
        "creepy atmospheric music",
        "tense dramatic background"
    ],

    'disgust': [
        "dark theme music",
        "distorted experimental tracks",
        "gritty industrial sounds",
        "unsettling glitch music",
        "raw underground beats"
    ],

    'surprise': [
        "energetic pop hits",
        "unexpected beat drops",
        "fast electro dance",
        "quirky fun music",
        "exciting upbeat edm"
    ]
}

    q = f"{queries.get(emotion,'songs')} {language} songs"

    try:
        results = sp.search(q=q, type='track', limit=limit)
    except:
        return [f"Sample offline {emotion} song"] * limit

    songs = [f"{t['name']} — {t['artists'][0]['name']} ({t['external_urls']['spotify']})"
             for t in results['tracks']['items']]
    return songs

# -------------------------------------------------------
# NLP PARSER
# -------------------------------------------------------
EMOTIONS = ["happy", "sad", "angry", "calm", "neutral", "fear", "disgust", "surprise","energy"]
LANGUAGES = ["english", "hindi", "marathi"]

def parse_user_input(text):
    text = text.lower()
    emo = next((e for e in EMOTIONS if e in text), "happy")
    lang = next((l for l in LANGUAGES if l in text), "english")
    return emo, lang

# -------------------------------------------------------
# MAIN PROGRAM
# -------------------------------------------------------
if __name__ == "__main__":
    MODEL = "saved_model.joblib"
    train_from_csv("metadata.csv", MODEL, use_cnn=False)
    clf, scaler, le, use_cnn = load_model(MODEL)
    if clf is None:
        print("⚠️ Train model first")
        exit()

    mode = input("Voice input (a) or Text input (t)? ").strip().lower()
    if mode == "a":
        if not HAS_SOUNDDEVICE:
            print("⚠️ sounddevice not found, use text mode")
            exit()
        fp = record_audio(3)
        f = extract_features(fp, use_cnn=use_cnn)
        f_scaled = f if use_cnn else scaler.transform([f])
        pred = clf.predict(f_scaled)[0]
        emotion = le.inverse_transform([pred])[0]

        lang = input("Language (english/hindi/marathi): ").lower().strip()
        songs = spotify_recommendations(emotion, lang)

        print(f"\n🧠 Emotion Detected: {emotion}")
        print("🎵 Recommended Songs:")
        for s in songs:
            print(" -", s)
    else:
        txt = input("Tell me your request: ")
        emotion, lang = parse_user_input(txt)
        songs = spotify_recommendations(emotion, lang)

        print(f"🧠 Emotion: {emotion}, Language: {lang}")
        for s in songs:
            print(" -", s)
