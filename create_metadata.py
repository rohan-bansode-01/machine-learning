import os
import csv

dataset_path = r"C:\ml_projects\project_folder"
csv_path = os.path.join(dataset_path, "metadata.csv")

rows = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split("-")
            if len(parts) > 2:
                emotion = parts[2].replace(".wav", "")
                rows.append([os.path.join(root, file), emotion])

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "emotion"])
    writer.writerows(rows)

print("✅ metadata.csv created at", csv_path)
