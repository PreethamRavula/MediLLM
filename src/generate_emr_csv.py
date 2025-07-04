import random
import csv
import string
from pathlib import Path

# Paths
CURRENT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = CURRENT_DIR.parent / "data" / "images"
OUTPUT_FILE = CURRENT_DIR.parent / "data" / "emr_records_softlabels.csv"

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
shared_symptoms = [
    "Mild cough and slight fever reported.",
    "General fatigue and throat irritation present.",
    "Breathing mildly labored during physical exertion.",
    "No major respiratory distress; mild wheezing noted.",
    "Occasional chest tightness reported.",
    "Vital signs mostly stable; slight variation in temperature.",
]

# Overlapping diagnosis clues
shared_diagnosis = [
    "Symptoms could relate to a range of viral infections.",
    "Presentation not distinctly matching any single infection.",
    "Further tests required to confirm diagnosis.",
    "Findings are borderline; clinical judgment advised.",
    "Observation warranted due to overlapping signs.",
    "Initial assessment inconclusive.",
]

# Noise sentences
neutral_noise = [
    "Patient is cooperative and alert.",
    "Dietary habits unremarkable.",
    "Hydration status normal.",
    "Follow-up advised if symptoms persist.",
    "No notable family medical history.",
    "No medications currently administered.",
]


def random_token():
    prefix = "ID"
    letters = "".join(random.choices(string.ascii_uppercase, k=2))
    digits = "".join(random.choices(string.digits, k=2))
    return f"{prefix}-{letters}{digits}"


def get_oxygen(label):
    # Soft blur across classes
    if label == "NORMAL":
        return random.randint(94, 100)
    elif label == "VIRAL PNEUMONIA":
        return random.randint(90, 96)
    else:
        return random.randint(87, 94)


def get_temp(label):
    if label == "NORMAL":
        return round(random.uniform(97.5, 99.0), 1)
    else:
        return round(random.uniform(98.8, 102.5), 1)


def get_age():
    return random.randint(18, 85)


def get_days():
    return random.randint(1, 10)


def build_emr(label, i):
    pid = random_token()
    age = f"{get_age()}-year-old"
    days = get_days()
    temp = get_temp(label)
    oxygen = get_oxygen(label)

    intro = f"Patient {pid}, a {age}, reports symptoms for {days} days."
    vitals = f"Temperature recorded at {temp}°F and SPO2 at {oxygen}%."

    # Shared symptoms + blurred logic
    body = [
        intro,
        random.choice(shared_symptoms),
        vitals,
        random.choice(shared_diagnosis),
    ]

    # Optionally inject a mild class-specific clue (with low probability)
    if random.random() < 0.3:
        if label == "COVID":
            body.append("Patient reports recent loss of taste.")
        elif label == "VIRAL PNEUMONIA":
            body.append("Chest X-ray shows scattered infiltrates.")
        elif label == "NORMAL":
            body.append("No active complaints at this time.")

    # Inject 1–2 noise sentences
    if random.random() < 0.8:
        body.insert(random.randint(1, len(body)), random.choice(neutral_noise))
    if random.random() < 0.5:
        body.insert(random.randint(1, len(body)), random.choice(neutral_noise))

    random.shuffle(body[1:])  # Keep intro in position 0
    return " ".join(body)


# Generate records
records = []
for label, img_dir in categories.items():
    image_files = sorted(
        [
            f
            for f in img_dir.glob("*")
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]
    )
    for i in range(SAMPLES_PER_CLASS):
        image_path = str(
            random.choice(image_files).relative_to(IMAGES_DIR.parent.parent)
        )
        text = build_emr(label, i)
        triage = triage_map[label]
        records.append([f"{label}-{i+1}", image_path, text, triage])

# Shuffle + write
random.shuffle(records)
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patient_id", "image_path", "emr_text", "triage_level"])
    writer.writerows(records)

print(f"✅ Softlabel EMR dataset generated at {OUTPUT_FILE}")
