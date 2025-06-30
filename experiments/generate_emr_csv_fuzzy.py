import random
import csv
from pathlib import Path

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = CURRENT_DIR.parent / "data" / "images"
OUTPUT_FILE = CURRENT_DIR.parent / "data" / "emr_records_fuzzy.csv"

# Sample size
SAMPLES_PER_CLASS = 300  # 900 total

# Categories and labels
categories = {
    "COVID": IMAGES_DIR / "COVID",
    "NORMAL": IMAGES_DIR / "NORMAL",
    "VIRAL PNEUMONIA": IMAGES_DIR / "VIRAL PNEUMONIA"
}

triage_map = {
    "COVID": "high",
    "NORMAL": "low",
    "VIRAL PNEUMONIA": "medium"
}

# --- Shared & ambiguous templates ---
noise_sentences = [
    "Follow-up scheduled for next week.",
    "Patient advised to maintain hydration and rest.",
    "No previous episodes of similar symptoms.",
    "Patient remains alert and oriented.",
    "Vitals are within acceptable ranges.",
    "No complications noted during assessment.",
    "Doctor recommends continued observation.",
    "Patient has no known drug allergies.",
    "Supportive care was initiated.",
    "Patient advised to avoid strenuous activity.",
    "Mild discomfort reported with no severe symptoms.",
    "Symptoms are self-limiting according to patient.",
    "No medication administered at this stage.",
    "Doctor recommends home rest and observation.",
    "Evaluation ongoing for possible infection."
]

ambiguous_templates = [
    "Mild fever noted. No cough. Patient recently traveled.",
    "Normal oxygen levels observed. Slight wheeze on auscultation.",
    "Patient reports chest discomfort but vitals are stable.",
    "No known exposure. Minor throat irritation present.",
    "Slight fatigue without other systemic symptoms."
]

# --- Vitals ---
def get_oxygen(label):
    ranges = {
        "COVID": (85, 94),
        "VIRAL PNEUMONIA": (88, 95),
        "NORMAL": (96, 99)
    }
    low, high = ranges[label]
    oxygen = random.randint(low - 1, high + 1)
    return min(100, max(80, oxygen))

def get_temp(label):
    if label == "NORMAL":
        temp = random.uniform(96.5, 99.0)
    else:
        temp = random.uniform(98.5, 104.0)
    return round(temp, 1)

def get_days(): return random.randint(1, 14)
def get_age(): return random.randint(18, 85)

# --- Build EMR ---
def build_emr(label, i):
    name = f"Patient-{label}-{i+1}"
    age = f"{get_age()}-year-old"
    days = get_days()
    temp = get_temp(label)
    oxygen = get_oxygen(label)

    # Shared symptoms across labels
    shared_symptoms = [
        f"{name} ({age}) reports dry cough and fatigue for {days} days.",
        f"{name} reports breathlessness. Temp recorded as {temp}°F.",
        f"{name} is experiencing low oxygen levels at {oxygen}%.",
        f"{name} complains of throat irritation and tiredness.",
        f"{name} has fever, but vitals are otherwise stable."
    ]

    # Label-specific diagnosis
    diagnosis = {
        "COVID": [
            "Findings suggest viral respiratory infection.",
            "Signs consistent with COVID-19 infection.",
            "Clinical features align with COVID diagnosis."
        ],
        "NORMAL": [
            "No signs of respiratory infection.",
            "Checkup results within normal limits.",
            "No abnormal findings detected."
        ],
        "VIRAL PNEUMONIA": [
            "X-ray shows patchy infiltrates.",
            "Clinical signs indicate viral pneumonia.",
            "Suspected viral origin of symptoms."
        ]
    }

    # Build full body
    emr = [random.choice(shared_symptoms), random.choice(diagnosis[label])]

    # Add ambiguity (~60%)
    if random.random() < 0.6:
        emr.insert(random.randint(0, len(emr)), random.choice(ambiguous_templates))

    # Add noise (~90%)
    if random.random() < 0.9:
        for _ in range(random.randint(1, 2)):
            emr.insert(random.randint(0, len(emr)), random.choice(noise_sentences))

    random.shuffle(emr)
    return " ".join(emr)

# --- Generate records ---
records = []
for label, img_dir in categories.items():
    files = sorted([f for f in img_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    for i in range(SAMPLES_PER_CLASS):
        image_path = str(random.choice(files).relative_to(IMAGES_DIR.parent.parent))
        emr_text = build_emr(label, i)
        triage = triage_map[label]
        pid = f"{label}-{i+1}"
        records.append([pid, image_path, emr_text, triage])

random.shuffle(records)

# --- Save to CSV ---
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patient_id", "image_path", "emr_text", "triage_level"])
    writer.writerows(records)

print(f"✅ Regenerated {len(records)} fuzzy EMR records at: {OUTPUT_FILE}")
