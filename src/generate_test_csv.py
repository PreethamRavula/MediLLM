import random
import csv
import string
from pathlib import Path

# Constants
SAMPLES_PER_CLASS = 10
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
TRAIN_CSV_PATH = PROJECT_ROOT / "data" / "emr_records.csv"
OUTPUT_CSV = PROJECT_ROOT / "test_samples.csv"
LABELS = ["COVID", "NORMAL", "VIRAL PNEUMONIA"]

# Labels to triage map
triage_map = {"COVID": "high", "NORMAL": "low", "VIRAL PNEUMONIA": "medium"}

alt_symptoms = [
    "The patient has noted intermittent chest pressure and occasional shortness of breath.",
    "A gradual onset of dry cough with mild respiratory discomfort has been documented.",
    "Reported complaints include mild fatigue and sporadic episodes of wheezing.",
    "Mild respiratory symptoms have progressed over several days.",
    "Episodes of throat irritation and general malaise observed.",
]

alt_diagnosis = [
    "Clinical features are suggestive of a nonspecific viral etiology.",
    "Diagnosis remains unclear pending further laboratory confirmation.",
    "Preliminary indicators fall into a diagnostic grey area.",
    "No definitive pattern observed; further evaluation is warranted.",
    "Presentation overlaps multiple pulmonary conditions.",
]

alt_noise = [
    "Patient remains oriented with stable hemodynamics.",
    "No remarkable family history or chronic illness reported.",
    "Nutritional intake and sleep patterns appear adequate.",
    "No prior admissions or surgical history disclosed.",
    "Standard precautions have been advised post-evaluation.",
]


def random_token():
    prefix = "TEST"
    letters = "".join(random.choices(string.ascii_uppercase, k=2))
    digits = "".join(random.choices(string.digits, k=2))
    return f"{prefix}-{letters}{digits}"


def get_oxygen(label):
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


def build_alt_emr(label):
    pid = random_token()
    age = f"{get_age()} years old"
    days = get_days()
    temp = get_temp(label)
    oxygen = get_oxygen(label)

    sent_intro = f"Patient {pid}, a {age} individual presented after experiencing symptoms for approximately {days} days."
    sent_vitals = f"Vital measurements include a body temperature of {temp}°F and an oxygen saturation level of {oxygen}%."

    body = [
        sent_intro,
        random.choice(alt_symptoms),
        sent_vitals,
        random.choice(alt_diagnosis),
    ]

    if random.random() < 0.3:
        if label == "COVID":
            body.append("Anosmia has been intermittently observed over recent days.")
        elif label == "VIRAL PNEUMONIA":
            body.append("Radiographic evidence reveals dispersed infiltrative patterns.")
        elif label == "NORMAL":
            body.append("There are currently no active complaints from the patient.")

    # inject 1-2 neutral clinical observations
    if random.random() < 0.9:
        body.insert(random.randint(1, len(body)), random.choice(alt_noise))
    if random.random() < 0.5:
        body.insert(random.randint(1, len(body)), random.choice(alt_noise))

    random.shuffle(body[1:])  # Keep the first sentence intact
    return " ".join(body)


def get_training_image_set():
    if not TRAIN_CSV_PATH.exists():
        raise FileNotFoundError(f"Training CSV not found at {TRAIN_CSV_PATH}")
    with open(TRAIN_CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        return set(row["image_path"].strip() for row in reader)


def generate_test_csv():
    training_images = get_training_image_set()
    records = []

    for label in LABELS:
        label_dir = IMAGE_DIR / label
        image_files = sorted([
            f for f in label_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ])
        unseen_images = [
            f for f in image_files
            if str(f.relative_to(PROJECT_ROOT)) not in training_images
        ]

        if len(unseen_images) < SAMPLES_PER_CLASS:
            raise ValueError(f"Not enough unseen images in {label_dir}."
                             f"Needed {SAMPLES_PER_CLASS}, found {len(unseen_images)}")
        sampled_images = random.sample(unseen_images, SAMPLES_PER_CLASS)

        for i, img_path in enumerate(sampled_images):
            relative_path = str(img_path.relative_to(PROJECT_ROOT))
            text = build_alt_emr(label)
            triage = triage_map[label]
            records.append([f"{label}-{i + 1}", text, relative_path, triage])

    random.shuffle(records)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "emr_text", "image_path", "triage_level"])
        writer.writerows(records)

    print(f"✅ test CSV file generated: {OUTPUT_CSV}")
    print(f"📦 Total samples: {len(records)} (10 per class)")


if __name__ == "__main__":
    generate_test_csv()
