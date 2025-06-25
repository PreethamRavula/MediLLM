import random
import csv
from pathlib import Path

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = CURRENT_DIR.parent / "data" / "images"
OUTPUT_FILE = CURRENT_DIR.parent / "data" / "emr_records_extended.csv"

# Sample size
SAMPLES_PER_CLASS = 1000 # 1000 * 3 = 3000 total

# Categories and labels
categories = {
    "COVID": IMAGES_DIR / "COVID",
    "NORMAL": IMAGES_DIR / "NORMAL",
    "VIRAL PNEUMONIA": IMAGES_DIR / "VIRAL PNEUMONIA"
}

# Triage mapping
triage_map = {
    "COVID": "high",
    "NORMAL": "low",
    "VIRAL PNEUMONIA": "medium"
}

# --- Noise Sentences ---
noise_sentences = [
    "Follow-up scheduled for next week.",
    "Patient advised to maintain hydration and rest.",
    "No previous episodes of similar symptoms.",
    "Patient remains alert and oriented.",
    "Vitals are within acceptable ranges.",
    "No complications noted during assessment.",
    "Patient has no known drug allergies.",
    "Doctor recommends continued observation.",
    "Supportive care was initiated.",
    "Patient advised to avoid strenuous activity."
]

# --- Vitals & Symptoms ---
def get_oxygen(label):
    return{
        "COVID": random.randint(85, 94),
        "VIRAL PNEUMONIA": random.randint(88, 95),
        "NORMAL": random.randint(96, 99)
    }[label]

def get_temp(label):
    return round(random.uniform(99.0, 103.5), 1) if label != "NORMAL" else round(random.uniform(97.0, 98.6), 1)

def get_days():
    return random.randint(1, 14)

# --- Templates ---
def build_emr(label):
    days = get_days()
    temp = get_temp(label)
    oxygen = get_oxygen(label)
    
    # Symptoms Pool
    symptoms = {
        "COVID": [
            f"Dry cough and fever for {days} days.",
            f"Loss of taste and smell observed.",
            f"Shortness of breath on exertion.",
            f"Persistent fatigue reported."
        ],
        "NORMAL": [
            "No active complaints.",
            "Patient in stable condition.",
            "Routine health check conducted.",
            "No respiratory symptoms reported."
        ],
        "VIRAL PNEUMONIA": [
            f"Dry cough lasting {days} days.",
            f"Fatigue and chest discomfort present.",
            f"Mild fever noted during checkup.",
            f"Shortness of breath at rest."
        ]
    }

    # Vitals Pool
    vitals = [
        f"Oxygen saturation is {oxygen}%.",
        f"SPO2 measured at {oxygen} percent.",
        f"Blood oxygen reads {oxygen}%.",
        f"Temperature recorded at {temp}°F."
    ]

    # Diagnosis Observations
    diagnosis = {
        "COVID": [
            "Findings suggest viral respiratory infection.",
            "Signs consistent with COVID-19 infection.",
            "Clinical features align with COVID diagnosis."
        ],
        "NORMAL": [
            "No signs of respiratory infection.",
            "No abnormal findings detected.",
            "Checkup results within normal limits."
        ],
        "VIRAL PNEUMONIA": [
            "X-ray shows patchy infiltrates.",
            "Suspected viral origin of symptoms.",
            "Clinical signs indicate viral pneumonia."
        ]
    }

    # Construct sentence pool
    sentences = [
        random.choice(symptoms[label]),
        random.choice(vitals),
        random.choice(diagnosis[label])
    ]

    # adding noise (~80% of cases)
    if random.random() < 0.8:
        noise = random.sample(noise_sentences, k=random.randint(1, 2))
        insert_at = random.randint(0, len(sentences))
        for n in noise:
            sentences.insert(insert_at, n)
    
    random.shuffle(sentences)
    return " ".join(sentences)

# Generate dataset
records = []
for label, img_dir in categories.items():
    image_files = sorted(img_dir.iterdir())
    for i in range(SAMPLES_PER_CLASS):
        patient_id = f"{label}-{i+1}"
        image_path = str(random.choice(image_files).relative_to(IMAGES_DIR.parent.parent))
        emr_text = build_emr(label)
        triage_level = triage_map[label]
        records.append([patient_id, image_path, emr_text, triage_level])

random.shuffle(records)

# Save to CSV
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patient_id", "image_path", "emr_text", "triage_level"])
    writer.writerows(records)

print(f"✅Generated {len(records)} varied EMR entries at {OUTPUT_FILE}")
