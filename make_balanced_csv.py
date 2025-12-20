import os
import pandas as pd
from glob import glob

# === 1️⃣ Set your dataset folder ===
DATASET_PATH = r"C:\ml_projects\project_folder"

# === 2️⃣ Emotion mapping for RAVDESS ===
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# === 3️⃣ Find all .wav files ===
files = glob(os.path.join(DATASET_PATH, "Actor_*", "*.wav"))

data = []
for f in files:
    fname = os.path.basename(f)
    parts = fname.split("-")
    if len(parts) < 7:
        continue  # skip malformed files
    emotion_id = parts[2]
    emotion = emotion_map.get(emotion_id)
    if not emotion:
        continue
    actor = os.path.basename(os.path.dirname(f))
    data.append([f, emotion, actor])

df = pd.DataFrame(data, columns=["filepath", "emotion", "actor"])

# === 4️⃣ Check counts before balancing ===
print("🎧 Emotion distribution before balancing:\n")
print(df["emotion"].value_counts())

# === 5️⃣ Balance all emotions equally ===
min_count = df["emotion"].value_counts().min()
balanced_df = df.groupby("emotion", group_keys=False).apply(
    lambda x: x.sample(min_count, random_state=42)
)

# === 6️⃣ Show balanced counts ===
print("\n✅ Emotion distribution after balancing:\n")
print(balanced_df["emotion"].value_counts())

# === 7️⃣ Save the balanced CSV ===
output_csv = os.path.join(DATASET_PATH, "metadata.csv")
balanced_df.to_csv(output_csv, index=False)
print(f"\n📁 Balanced CSV saved to: {output_csv}")
