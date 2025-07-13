import os
import re
import csv
import sys
import pytest
from pathlib import Path

# Add src/ to path so we can import from it
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from generate_emr_csv import generate_dataset, OUTPUT_FILE

# Determine if running in CI
IS_CI = os.getenv("CI", "false").lower() == "true"

# Paths
DATA_DIR = BASE_DIR / "data"
DUMMY_IMAGES_DIR = DATA_DIR / "dummy_images"
REAL_IMAGES_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / ("test_emr_records.csv" if IS_CI else OUTPUT_FILE)

# Constants
EXPECTED_COLUMNS = ["patient_id", "image_path", "emr_text", "triage_level"]
EXPECTED_CLASSES = ["low", "medium", "high"]
EXPECTED_SAMPLES_PER_CLASS = 3 if IS_CI else 300

AMBIGUOUS_PHRASES = [
    "Symptoms could relate to a range of viral infections.",
    "Presentation not distinctly matching any single infection.",
    "Further tests required to confirm diagnosis.",
    "Findings are borderline; clinical judgment advised.",
    "Observation warranted due to overlapping signs.",
    "Initial assessment inconclusive.",
]

SHARED_SYMPTOMS = [
    "Mild cough and slight fever reported.",
    "General fatigue and throat irritation present.",
    "Breathing mildly labored during physical exertion.",
    "No major respiratory distress; mild wheezing noted.",
    "Occasional chest tightness reported.",
    "Vital signs mostly stable; slight variation in temperature.",
]

NOISE_SENTENCES = [
    "Patient is cooperative and alert.",
    "Dietary habits unremarkable.",
    "Hydration status normal.",
    "Follow-up advised if symptoms persist.",
    "No notable family medical history.",
    "No medications currently administered.",
]


@pytest.fixture(scope="module", autouse=True)
def generate_csv_for_test():
    image_dir = DUMMY_IMAGES_DIR if IS_CI else REAL_IMAGES_DIR
    generate_dataset(image_dir_override=image_dir, output_path_override=CSV_PATH)


def test_csv_exists():
    assert CSV_PATH.exists(), f"CSV file not found at: {CSV_PATH}"


def test_csv_structure():
    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    assert set(header) == set(EXPECTED_COLUMNS), "CSV columns mismatch"


def test_total_and_per_class_counts():
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    expected_total = EXPECTED_SAMPLES_PER_CLASS * len(EXPECTED_CLASSES)
    assert len(rows) == expected_total

    counts = {"low": 0, "medium": 0, "high": 0}
    for row in rows:
        counts[row["triage_level"]] += 1

    assert all(c == EXPECTED_SAMPLES_PER_CLASS for c in counts.values)


def test_patient_id_format_and_uniqueness(load_emr_csv):
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        ids = [row["patient_id"] for row in reader]
        assert len(ids) == len(set(ids)), "Duplicate patient IDs found"
        pattern = re.compile(r"^ID-[A-Z]{2}\d{2}$")
        for pid in ids:
            assert pattern.match(pid), f"Invalid patient ID format: {pid}"


def test_emr_text_quality():
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["emr_text"]
            assert (
                isinstance(text, str) and len(text.split()) > 10
            ), "EMR text too short or malformed"
            assert "Temperature" in text and "SPO2" in text, "Vitals info missing"


def test_image_path_format():
    expected_path = DUMMY_IMAGES_DIR.relative_to(BASE_DIR) if IS_CI else REAL_IMAGES_DIR.relative_to(BASE_DIR)
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["image_path"]
            assert path.startswith(expected_path), f"Image path should start with '{expected_path}', got: {path}"
            assert path.endswith((".jpg", ".jpeg", ".png")), f"Invalid image path: {path}"


def test_ambiguous_and_noise_injection(load_emr_csv):
    ambiguous_hits = 0
    symptom_hits = 0
    noise_hits = 0

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["emr_text"]
            if any(phrase in text for phrase in AMBIGUOUS_PHRASES):
                ambiguous_hits += 1
            if any(symptom in text for symptom in SHARED_SYMPTOMS):
                symptom_hits += 1
            if any(noise in text for noise in NOISE_SENTENCES):
                noise_hits += 1

    assert ambiguous_hits > 800, "Ambiguous phrases missing in too many EMRs"
    assert symptom_hits > 800, "Shared symptom clues underrepresented"
    assert noise_hits > 700, "Too few EMRs contain noise sentences"


def test_label_validity():
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert (
                row["triage_level"] in EXPECTED_CLASSES
            ), f"Invalid label: {row['triage_level']}"


def test_no_empty_fields():
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                assert val.strip() != "", f"Empty field found for {key}"
