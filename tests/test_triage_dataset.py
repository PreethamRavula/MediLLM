import os
import sys
import pytest
import torch
import pandas as pd
from pathlib import Path

# Setup path to src/
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from triage_dataset import TriageDataset

# Detect CI environment
IS_CI = os.getenv("CI", "false").lower() == "true"

# Paths
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / ("test_emr_records.csv" if IS_CI else "emr_records.csv")
IMAGE_DIR = DATA_DIR / ("dummy_images" if IS_CI else "images")
EXPECTED_SAMPLES_PER_CLASS = 3 if IS_CI else 300
EXPECTED_TOTAL = 3 * 3 if IS_CI else 300 * 3  # 3 classes


@pytest.mark.parametrize("mode", ["text", "image", "multimodal"])
def test_dataset_loading(mode):
    kwargs = {"csv_file": CSV_PATH, "mode": mode}
    if mode in ["image", "multimodal"]:
        kwargs["image_base_dir"] = IMAGE_DIR

    dataset = TriageDataset(**kwargs)

    # Check dataset length
    assert len(dataset) == EXPECTED_TOTAL, f"Expected {EXPECTED_TOTAL} records in the dataset"

    # Check one sample
    sample = dataset[0]

    if mode in ["text", "multimodal"]:
        assert "input_ids" in sample, "Missing input_ids in text/multimodal mode"
        assert (
            "attention_mask" in sample
        ), "Missing attention_mask in text/multimodal mode"
        assert sample["input_ids"].shape[0] == 128, "Incorrect token length"

    if mode in ["image", "multimodal"]:
        assert "image" in sample, "Missing image in image/multimodal mode"
        assert isinstance(sample["image"], torch.Tensor), "Image not a tensor"
        assert sample["image"].shape[1:] == (224, 224), "Incorrect image size"

    assert "label" in sample, "Missing label"
    assert sample["label"].item() in [0, 1, 2], "Invalid label value"


def test_missing_image_raises_error(tmp_path):
    # Create a temporary CSV file with an invalid image path
    fake_csv = tmp_path / "fake_emr_records.csv"
    fake_df = pd.DataFrame([{
        "patient_id": "ID-XX99",
        "image_path": "data/images/NORMAL/non_existent_image.jpg",
        "emr_text": "Patient ID-XX99 reports symptoms. Temperature recorded at 98.6°F and SPO2 at 97%.",
        "triage_level": "low"
    }])
    fake_df.to_csv(fake_csv, index=False)

    # Instantiate the dataset in image mode
    dataset = TriageDataset(csv_file=fake_csv, mode="image", image_base_dir=IMAGE_DIR)

    # Expect a FileNotFoundError when trying to access the missing image
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        _ = dataset[0]
