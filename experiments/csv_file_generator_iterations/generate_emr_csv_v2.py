import random
import csv
import string
from pathlib import Path

# Paths
CURRENT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = CURRENT_DIR.parent / "data" / "images"
OUTPUT_FILE = CURRENT_DIR.parent / "data" / "emr_records_richfuzzy.csv"

# Label to triage
triage_map = {"COVID": "high", "NORMAL": "low", "VIRAL PNEUMONIA": "medium"}
SAMPLES_PER_CLASS = 300

# Folders
categories = {
    "COVID": IMAGES_DIR / "COVID",
    "NORMAL": IMAGES_DIR / "NORMAL",
    "VIRAL PNEUMONIA": IMAGES_DIR / "VIRAL PNEUMONIA",
}

# Shared ambiguous templates
ambiguous_phrases = [
    "Slight throat irritation without systemic symptoms.",
    "Mild dyspnea but normal vitals.",
    "Minor dry cough reported, patient stable.",
    "Chest X-ray inconclusive.",
    "No recent exposure or travel noted.",
    "Intermittent headache without fever.",
]

# Noise sentences
neutral_noise = [
    "Patient is cooperative and alert.",
    "Dietary habits unremarkable.",
    "Follow-up recommended if symptoms persist.",
    "Hydration status is normal.",
    "No family history of chronic illness.",
    "Patient expresses concern about possible flu.",
]


# ---Patient random token genrator ---
def random_token():
    prefix = "ID"
    letters = "".join(random.choices(string.ascii_uppercase, k=2))
    digits = "".join(random.choices(string.digits, k=2))
    return f"{prefix}-{letters}{digits}"


# Vitals (blurred)
def get_oxygen(label):
    base = {"COVID": (85, 94), "VIRAL PNEUMONIA": (89, 96), "NORMAL": (96, 99)}
    min_, max_ = base[label]
    return min(100, max(80, random.randint(min_ - 1, max_ + 1)))


def get_temp(label):
    if label == "NORMAL":
        min_, max_ = 97.0, 98.5
    else:
        min_, max_ = 99.0, 103.0
    return round(random.uniform(min_ - 0.6, max_ + 0.6), 1)


def get_age():
    return random.randint(18, 85)


def get_days():
    return random.randint(1, 10)


# EMR generator
def build_emr(label, i):
    patient_id = random_token()
    age = f"{get_age()}-year-old"
    oxygen = get_oxygen(label)
    temp = get_temp(label)
    days = get_days()

    general_intro = (
        f"Patient {patient_id}, a {age}, presents with symptoms for {days} days."
    )
    vitals = f"Temperature recorded at {temp}°F, SPO2 levels at {oxygen}%."

    # Label-specific (but fuzzy) symptoms
    symptoms = {
        "COVID": [
            "Complains of fatigue and shortness of breath.",
            "Dry cough with mild fever noted.",
        ],
        "NORMAL": [
            "No major complaints; here for general checkup.",
            "Reports good health, no active issues.",
        ],
        "VIRAL PNEUMONIA": [
            "Persistent cough and mild fever observed.",
            "Slight wheezing with chest tightness.",
        ],
    }

    diagnosis = {
        "COVID": ["Viral etiology suspected.", "COVID infection not ruled out."],
        "NORMAL": ["Unlikely presence of infection.", "Clinical impression is benign."],
        "VIRAL PNEUMONIA": [
            "Signs may indicate atypical pneumonia.",
            "Possible viral infection of lower tract.",
        ],
    }

    body = [
        general_intro,
        random.choice(symptoms[label]),
        vitals,
        random.choice(diagnosis[label]),
    ]

    # Inject 1–2 ambiguous or neutral sentences
    if random.random() < 0.8:
        body.insert(random.randint(1, len(body)), random.choice(ambiguous_phrases))
    if random.random() < 0.7:
        body.insert(random.randint(1, len(body)), random.choice(neutral_noise))

    random.shuffle(body[1:])
    return " ".join(body)


# Generate records
records = []
for label, img_dir in categories.items():
    image_files = sorted(
        [f for f in img_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )
    for i in range(SAMPLES_PER_CLASS):
        image_path = str(
            random.choice(image_files).relative_to(IMAGES_DIR.parent.parent)
        )
        text = build_emr(label, i)
        triage = triage_map[label]
        records.append([f"{label}-{i + 1}", image_path, text, triage])

# Shuffle + Write
random.shuffle(records)
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patient_id", "image_path", "emr_text", "triage_level"])
    writer.writerows(records)

print(f"✅ Rich fuzzy EMR dataset saved at {OUTPUT_FILE}")
