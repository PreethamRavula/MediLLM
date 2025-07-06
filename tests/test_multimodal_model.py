import sys
import os
import torch
import pytest
from transformers import AutoTokenizer


# Add repo root to the sys.path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.multimodal_model import MediLLMModel

BATCH_SIZE = 2
SEQ_LEN = 128
IMAGE_SIZE = (3, 224, 224)
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)


@pytest.fixture
def dummy_inputs():
    text_batch = ["Patient reports mild cough and fever."] * BATCH_SIZE
    encoding = tokenizer(
        text_batch,
        padding="max_length",
        truncation=True,
        max_length=SEQ_LEN,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "image": torch.randn(BATCH_SIZE, *IMAGE_SIZE),
    }


def test_text_only(dummy_inputs):
    model = MediLLMModel(mode="text")
    model.eval()
    outputs = model(
        input_ids=dummy_inputs["input_ids"],
        attention_mask=dummy_inputs["attention_mask"],
    )
    assert outputs.shape == (BATCH_SIZE, 3), "Incorrect output shape for text-only mode"


def test_image_only(dummy_inputs):
    model = MediLLMModel(mode="image")
    model.eval()
    outputs = model(image=dummy_inputs["image"])
    assert outputs.shape == (
        BATCH_SIZE,
        3,
    ), "Incorrect output shape for image-only mode"


def test_multimodal(dummy_inputs):
    model = MediLLMModel(mode="multimodal")
    model.eval()
    outputs = model(
        input_ids=dummy_inputs["input_ids"],
        attention_mask=dummy_inputs["attention_mask"],
        image=dummy_inputs["image"],
    )
    assert outputs.shape == (
        BATCH_SIZE,
        3,
    ), "Incorrect output shape for multimodal mode"
