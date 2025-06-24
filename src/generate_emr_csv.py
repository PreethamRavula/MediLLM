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

# EMR templates for each class (15 diverse ones each)
templates = {
    "COVID": [
        "<NAME> (age <AGE>) complains of fever and cough for {days} days. Oxygen level is {oxygen}%.",
        "Loss of taste and smell reported. Cough persists for {days} days. SPO2: {oxygen}%.",
        "Experiencing dry cough, fever. Temperature: {temp}°F. <NAME> shows respiratory strain.",
        "Rapid breathing and low oxygen detected. SPO2: {oxygen}%, Temp: {temp}°F.",
        "History of recent exposure. Presents with mild fever and persistent dry cough.",
        "<NAME> reports fatigue and shortness of breath. Onset {days} days ago.",
        "High temperature of {temp}°F. SPO2 dropped to {oxygen}%. Cough continues.",
        "Mild chest pain during coughing. Loss of appetite. Temperature and vitals fluctuating.",
        "Symptoms consistent with viral infection. O2 saturation: {oxygen}%.",
        "X-ray suggests diffuse opacities. SPO2 {oxygen}%, Temp {temp}°F.",
        "Subject fails to recognize smells. Blood oxygen level below 94%.",
        "Persistent fatigue. SOB while walking. Fever {temp}°F.",
        "Ongoing fever, chest congestion, and dry cough. SPO2 dropping.",
        "Recent travel history. Complains of throat pain and dry cough.",
        "Elevated temperature and low energy. Diagnosed positive with antigen test."
    ],
    "NORMAL": [
        "Routine physical exam. <NAME> reports no issues. All vitals normal.",
        "No complaints during checkup. SPO2: {oxygen}%, Temp: {temp}°F.",
        "<NAME> is asymptomatic. Lung auscultation clear. Heart rate steady.",
        "Checkup for occupational clearence. Normal findings.",
        "Vitals normal. Patient denies fever, cough, or fatigue.",
        "General wellness check. No signs of infection.",
        "No abnormalities detected. Stable condition.",
        "<NAME> visited for follow-up. No respiratory symptoms present.",
        "X-ray reveals no opacities. SPO2 at {oxygen}%.",
        "Patient in good health. BMI and vitals within range.",
        "Denies any contact with infected individuals.",
        "Clear medical history. No medication use reported.",
        "Annual exam. Temp {temp}°F. lungs sound normal while breathing.",
        "No clinical findings. Breathing unlaboured.",
        "<NAME> presents for routine screening. All parameters stable."
    ],
    "VIRAL PNEUMONIA": [
        "Dry cough for {days} days. Mild fever noted. SPO2: {oxygen}%, Temp: {temp}°F.",
        "Chest pain on deep breaths. Signs of lung inflammation.",
        "<NAME> reports headache, fatigue, and cough. Vital signs fluctuating.",
        "X-ray shows patchy infiltrates. Low-grade fever.",
        "Wheezing noted. Fatigue and congestion present.",
        "Cough has persisted since {days} days. No known exposure.",
        "Symptoms consistent with viral pneumonia. Temp: {temp}°F.",
        "Breath sounds diminished. Mild respiratory distress.",
        "Crackles ausculated bilaterally. Oxygen at {oxygen}%.",
        "Recovery from a recent flu-like illness. Moderate symptoms remain.",
        "Shortness of breath with fatigue. Lungs show minor opacity.",
        "Fever peaks at {temp}°F. Mild SOB on exertion.",
        "Patient <NAME> experiencing night sweats, slight chest congestion.",
        "Lung X-ray suggests viral origin. Oxygen: {oxygen}%.",
        "Cough with low energy. Symptoms persistent past {days} days."
    ]
}

# Class-based logic for O2 and temperature
def get_oxygen(label):
    if label == "COVID":
        return random.randint(85, 94)
    elif label == "VIRAL PNEUMONIA":
        return random.randint(88, 95)
    elif label == "NORMAL":
        return random.randint(96, 99)

def get_temp(label):
    if label == "NORMAL":
        return round(random.uniform(97.0, 98.6), 1)
    else:
        return round(random.uniform(99.0, 103.5), 1)

# EMR generator function
def generate_emr(label):
    days = random.randint(1, 14)
    temp = get_temp(label)
    oxygen = get_oxygen(label)
    template = random.choice(templates[label])
    return template.replace("<NAME>", "<NAME>").replace("<AGE>", "<AGE>").format(
        days=days, temp=temp, oxygen=oxygen
    )

# Generate dataset
records = []
for label, img_dir in categories.items():
    image_files = sorted(img_dir.iterdir())
    for i in range(SAMPLES_PER_CLASS):
        patient_id = f"{label}-{i+1}"
        image_path = str(random.choice(image_files).relative_to(IMAGES_DIR.parent.parent))
        emr_text = generate_emr(label)
        triage_level = triage_map[label]
        records.append([patient_id, image_path, emr_text, triage_level])

random.shuffle(records)

# Save to CSV
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patient_id", "image_path", "emr_text", "triage_level"])
    writer.writerows(records)

print(f"✅Generated {len(records)} medically consistent EMR entries in {OUTPUT_FILE}")
