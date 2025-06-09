import random
import csv
from pathlib import Path

# Get path to current script
CURRENT_DIR = Path(__file__).resolve().parent

# Go up to the project root and then to data/Images
IMAGES_DIR = CURRENT_DIR.parent / "data" / "images"

# Go to the Project root and then to data
OUTPUT_FILE = CURRENT_DIR.parent / "data" / "emr_records.csv"

# Directory paths
categories = {
    "COVID": IMAGES_DIR / "COVID",
    "NORMAL": IMAGES_DIR / "NORMAL",
    "VIRAL PNEUMONIA": IMAGES_DIR / "VIRAL PNEUMONIA"
}

# EMR templates for each class
templates = {
    "COVID": [
        "The patient presents with a dry cough, fever, and shortness of breath. Symptoms began 5 days ago.",
        "Progressive difficulty in breathing. Oxygen saturation is below the normal range.",
        "The patient reports loss of taste and smell, with a persistent cough."
    ],
    "NORMAL": [
        "Routine checkup with no abnormal findings. The patient denies cough or chest pain.",
        "Clear lungs on auscultation. No signs of infection. Chest x-ray was unremarkable.",
        "No complaints. Normal vitals and physical exam."
    ],
    "VIRAL PNEUMONIA": [
        "Mild fever, chest tightness, and dry cough for the past 3 days. Oxygen levels are normal.",
        "Crackles are auscultated in the lower lobes. The patient presents with fatigue and mild respiratory distress.",
        "The X-ray shows patchy infiltrates in the lungs. The patient is recovering from a recent viral infection."
    ]
}

# Triage mapping
triage_map = {
    "COVID": "high",
    "NORMAL": "low",
    "VIRAL PNEUMONIA": "medium"
}

# Generate CSV rows
records = []
for label, dir_path in categories.items():
    for i, img_path in enumerate(sorted(dir_path.iterdir())[:30], start=1):
        patient_id = f"{label}-{i}"
        image_path = f"images/{label}/{img_path.name}"
        emr_text = random.choice(templates[label])
        triage_level = triage_map[label]
        records.append([patient_id, image_path, emr_text, triage_level])

# Write to CSV
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patient_id", "image_path", "emr_text", "triage_level"])
    writer.writerows(records)

print(f"✅Generated {len(records)} EMR entries in {OUTPUT_FILE}")
