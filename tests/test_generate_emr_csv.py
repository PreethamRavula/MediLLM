import os
import csv
import sys
import pytest
from collections import Counter

# Add repo root to the sys.path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.generate_emr_csv import generate_dataset, OUTPUT_FILE


CSV_PATH = OUTPUT_FILE
EXPECTED_CLASSES = {"low", "medium", "high"}
EXPECTED_COLUMNS = ["patient_id", "image_path", "emr_text", "triage_level"]
EXPECTED_SAMPLES_PER_CLASS = 300

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


def test_dataset_generation_runs():
    generate_dataset()
    assert CSV_PATH.exists(), "CSV file should be generated"
    with open(OUTPUT_FILE, "r") as f:
        lines = f.readlines()
    assert len(lines) > 1  # Header + Content


@pytest.fixture(scope="module")
def load_emr_csv():
    assert CSV_PATH.exists(), f"CSV file not found at: {CSV_PATH}"
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def test_csv_structure(load_emr_csv):
    row = load_emr_csv[0]
    assert set(row.keys()) == set(EXPECTED_COLUMNS), "CSV columns mismatch"


def test_total_and_per_class_counts(load_emr_csv):
    assert len(load_emr_csv) == 900, "Total records should be 900"
    counts = Counter(row["triage_level"] for row in load_emr_csv)
    for cls in EXPECTED_CLASSES:
        assert counts[cls] == EXPECTED_SAMPLES_PER_CLASS, (
            f"{cls} count mismatch"
        )


def test_patient_id_format_and_uniqueness(load_emr_csv):
    ids = [row["patient_id"] for row in load_emr_csv]
    assert all(id and "-" in id for id in ids), "Malformed patient IDs found"
    assert len(set(ids)) == 900, "Duplicate patient IDs found"


def test_emr_text_quality(load_emr_csv):
    for row in load_emr_csv:
        text = row["emr_text"]
        assert (
            isinstance(text, str) and len(text.split()) > 10
        ), "EMR text too short or malformed"
        assert "Temperature" in text and "SPO2" in text, "Vitals info missing"


def test_image_path_format(load_emr_csv):
    for row in load_emr_csv:
        path = row["image_path"]
        assert path.endswith((".jpg", ".jpeg", ".png")), (
            f"Invalid image path: {path}"
        )


def test_ambiguous_and_noise_injection(load_emr_csv):
    ambiguous_hits = 0
    symptom_hits = 0
    noise_hits = 0

    for row in load_emr_csv:
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


def test_label_validity(load_emr_csv):
    for row in load_emr_csv:
        assert (
            row["triage_level"] in EXPECTED_CLASSES
        ), f"Invalid label: {row['triage_level']}"


def test_no_empty_fields(load_emr_csv):
    for row in load_emr_csv:
        for col in EXPECTED_COLUMNS:
            assert row[col].strip(), f"Empty field found in colum '{col}'"
