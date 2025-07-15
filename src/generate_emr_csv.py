import os
import random
import csv
import string
from pathlib import Path

# Detect CI environment
IS_CI = os.getenv("CI", "false").lower() == "true"

# Set number of samples accordingly
SAMPLES_PER_CLASS = 3 if IS_CI else 300  # Reduced for CI to speed up tests

# Paths
CURRENT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = CURRENT_DIR.parent / "data" / "images"  # Absolute path of images folder
OUTPUT_FILE = CURRENT_DIR.parent / "data" / "emr_records.csv"

# Label to triage
triage_map = {"COVID": "high", "NORMAL": "low", "VIRAL PNEUMONIA": "medium"}

# Shared ambiguous templates
shared_symptoms = [
    "Mild cough and slight fever reported.",
    "General fatigue and throat irritation present.",
    "Breathing mildly labored during physical exertion.",
    "No major respiratory distress; mild wheezing noted.",
    "Occasional chest tightness reported.",
    "Vital signs mostly stable; slight variation in temperature.",
]

# Overlapping diagnosis clues to add ambiguity
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
def generate_dataset(image_dir_override=None, output_path_override=None):
    root_image_dir = image_dir_override or IMAGES_DIR
    output_file = output_path_override or OUTPUT_FILE

    # Folders
    categories = {
        "COVID": root_image_dir / "COVID",  # Absolute path of Image labels
        "NORMAL": root_image_dir / "NORMAL",
        "VIRAL PNEUMONIA": root_image_dir / "VIRAL PNEUMONIA",
    }

    records = []
    for label, img_dir in categories.items():
        image_files = sorted(
            [
                f
                for f in img_dir.glob("*")
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
            ]
        )
        if not image_files:
            raise FileNotFoundError(
                f"No images found in {img_dir}. Folder contents: {list(img_dir.iterdir())}")

        for i in range(SAMPLES_PER_CLASS):
            image_path = str(
                random.choice(image_files).relative_to(root_image_dir.parent)  # path of image respective to the project root
            )
            text = build_emr(label, i)
            triage = triage_map[label]
            records.append([f"{label}-{i + 1}", image_path, text, triage])

    # Shuffle + write
    random.shuffle(records)
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "image_path", "emr_text", "triage_level"])
        writer.writerows(records)

    print(f"✅ EMR dataset generated at {output_file}")


if __name__ == "__main__":
    generate_dataset(image_dir_override=IMAGES_DIR, output_path_override=OUTPUT_FILE)
