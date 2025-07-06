import os
import sys
import pytest
import torch

base_dir = os.path.dirname(os.path.dirname(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.triage_dataset import TriageDataset

# Path to CSV and example image should match the local structure
CSV_PATH = os.path.join(base_dir, "data", "emr_records.csv")


@pytest.mark.parametrize("mode", ["text", "image", "multimodal"])
def test_dataset_loading(mode):
    dataset = TriageDataset(csv_file=CSV_PATH, mode=mode)

    # Check dataset length
    assert len(dataset) == 900, "Expected 900 records in the dataset"

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
