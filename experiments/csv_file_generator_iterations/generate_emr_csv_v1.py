import random
import csv
from pathlib import Path

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = CURRENT_DIR.parent / "data" / "images"
OUTPUT_FILE = CURRENT_DIR.parent / "data" / "emr_records_extended.csv"

# Sample size
SAMPLES_PER_CLASS = 300  # 300 * 3 = 900 total

# Categories and labels
categories = {
    "COVID": IMAGES_DIR / "COVID",
    "NORMAL": IMAGES_DIR / "NORMAL",
    "VIRAL PNEUMONIA": IMAGES_DIR / "VIRAL PNEUMONIA",
}

# Triage mapping
triage_map = {"COVID": "high", "NORMAL": "low", "VIRAL PNEUMONIA": "medium"}

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
    "Patient advised to avoid strenuous activity.",
    "No complications noted during assessment",
    "No prior history of respiratory illness.",
    "Mild discomfort reported with no severe symptoms.",
    "Symptoms are self-limiting according to patient.",
    "Patient remains alert and cooperative.",
    "No medication administered at this stage.",
    "Doctor recommends home resr and observation.",
    "Evaluation ongoing for possible infection.",
]

# --- ambiguity sentences ---
ambiguous_templates = [
    "Mild fever noted. No cough. Patient recently traveled.",
    "Normal oxygen levels observed. Slight wheeze on auscultation.",
    "Patient reports chest discomfort but vitals are stable.",
    "No known exposure. Minor throat irritation present.",
    "Slight fatigue without other systemic symptoms.",
]

# --- Vitals & Symptoms ---


def get_oxygen(label):
    base_ranges = {"COVID": (85, 94), "VIRAL PNEUMONIA": (88, 95), "NORMAL": (96, 99)}
    base_min, base_max = base_ranges[label]
    # Apply + or - 1 blur, clamping between 80 and 100
    oxygen = random.randint(base_min - 1, base_max + 1)
    return min(100, max(80, oxygen))


def get_temp(label):
    if label == "NORMAL":
        base_min, base_max = 97.0, 98.6
    else:
        base_min, base_max = 99.0, 103.5

    # Apply + or - 0.5°F blur and clamp between 95-105°F
    temp = random.uniform(base_min - 0.5, base_max + 0.5)
    return round(min(105.0, max(95.0, temp)), 1)


def get_days():
    return random.randint(1, 14)


def get_age():
    return random.randint(18, 80)


# --- Templates ---
def build_emr(label, i):
    name = f"Patient-{label}-{i + 1}"
    age = f"{get_age()}-year-old"
    days = get_days()
    temp = get_temp(label)
    oxygen = get_oxygen(label)
    # Symptoms Pool
    symptoms = {
        "COVID": [
            f"{name} ({age}) reports fatigue and dry cough for {days} days.",
            f"{name} complains of shortness of breath and fever of {temp}°F.",
            f"{name} reports loss of taste. SPO2 at {oxygen}%.",
        ],
        "NORMAL": [
            f"{name} ({age}) presents for routine check-up. Vitals stable.",
            f"{name} shows no respiratory distress. Oxygen at {oxygen}%.",
            f"{name} denies any recent illness. Temperature is {temp}°F.",
        ],
        "VIRAL PNEUMONIA": [
            f"{name} ({age}) complains of dry cough for {days} days.",
            f"{name} experiencing low-grade fever and SPO2 at {oxygen}%.",
            f"{name} reports breathlessness. X-ray indicates mild infiltrates.",
        ],
    }

    # Diagnosis Observations
    diagnosis = {
        "COVID": [
            "Findings suggest viral respiratory infection.",
            "Signs consistent with COVID-19 infection.",
            "Clinical features align with COVID diagnosis.",
        ],
        "NORMAL": [
            "No signs of respiratory infection.",
            "No abnormal findings detected.",
            "Checkup results within normal limits.",
        ],
        "VIRAL PNEUMONIA": [
            "X-ray shows patchy infiltrates.",
            "Suspected viral origin of symptoms.",
            "Clinical signs indicate viral pneumonia.",
        ],
    }

    # Construct sentence pool
    body = [random.choice(symptoms[label]), random.choice(diagnosis[label])]

    # adding ambiguous cases randomly (~70% of cases)
    if random.random() < 0.7:
        body.insert(random.randint(0, len(body)), random.choice(ambiguous_templates))

    # adding noise to 90% of cases
    if random.random() < 0.9:
        for _ in range(random.randint(1, 2)):
            body.insert(random.randint(0, len(body)), random.choice(noise_sentences))
    random.shuffle(body)
    return " ".join(body)


# Generate dataset
records = []
for label, img_dir in categories.items():
    valid_exts = [".png", ".jpg", ".jpeg"]
    image_files = sorted(
        [f for f in img_dir.glob("*") if f.suffix.lower() in valid_exts]
    )
    for i in range(SAMPLES_PER_CLASS):
        patient_id = f"{label}-{i + 1}"
        image_path = str(
            random.choice(image_files).relative_to(IMAGES_DIR.parent.parent)
        )
        emr_text = build_emr(label, i)
        triage_level = triage_map[label]
        records.append([patient_id, image_path, emr_text, triage_level])

random.shuffle(records)

# Save to CSV
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patient_id", "image_path", "emr_text", "triage_level"])
    writer.writerows(records)

print(f"✅Generated {len(records)} EMR records in {OUTPUT_FILE}")
