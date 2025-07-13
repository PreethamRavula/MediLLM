import sys
import torch
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# Add repo root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.multimodal_model import MediLLMModel

BATCH_SIZE = 2
SEQ_LEN = 128
IMAGE_SIZE = (3, 224, 224)


@pytest.fixture
def dummy_inputs():
    return {
        "input_ids": torch.randint(0, 30522, (BATCH_SIZE, SEQ_LEN)),  # dummy token IDs
        "attention_mask": torch.ones(BATCH_SIZE, SEQ_LEN),
        "image": torch.randn(BATCH_SIZE, *IMAGE_SIZE),
    }


@patch("src.multimodal_model.AutoModel.from_pretrained")
@patch("src.multimodal_model.timm.create_model")
def test_text_only(mock_create_model, mock_auto_model, dummy_inputs):
    # Mock text encoder
    mock_text_encoder = MagicMock()
    mock_text_encoder.config.hidden_size = 768
    mock_text_encoder.return_value = MagicMock(
        last_hidden_state=torch.randn(BATCH_SIZE, SEQ_LEN, 768)
    )
    mock_auto_model.return_value = mock_text_encoder

    model = MediLLMModel(mode="text")
    model.eval()
    outputs = model(
        input_ids=dummy_inputs["input_ids"],
        attention_mask=dummy_inputs["attention_mask"]
    )

    assert outputs.shape == (BATCH_SIZE, 3)
    probs = torch.softmax(outputs, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-5)


@patch("src.multimodal_model.Automodel.from_pretrained")
@patch("src.multimodal_model.timm.create_model")
def test_image_only(mock_create_model, mock_auto_model, dummy_inputs):
    # Mock image encoder
    mock_image_encoder = MagicMock()
    mock_image_encoder.num_features = 2048
    mock_image_encoder.return_value = torch.randn(BATCH_SIZE, 2048)
    mock_create_model.return_value = mock_image_encoder

    model = MediLLMModel(mode="image")
    model.eval()
    outputs = model(image=dummy_inputs["image"])

    assert outputs.shape == (BATCH_SIZE, 3)
    probs = torch.softmax(outputs, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-5)


@patch("src.multimodal_model.AutoModel.from_pretrained")
@patch("src.multimodal_model.timm.create_model")
def test_multimodal(mock_create_model, mock_auto_model, dummy_inputs):
    # Mock text encoder
    mock_text_encoder = MagicMock()
    mock_text_encoder.config.hidden_size = 768
    mock_text_encoder.return_value = MagicMock(
        last_hidden_state=torch.randn(BATCH_SIZE, SEQ_LEN, 768)
    )
    mock_auto_model.return_value = mock_text_encoder

    # Mock image encoder
    mock_image_encoder = MagicMock()
    mock_image_encoder.num_features = 2048
    mock_image_encoder.return_value = torch.randn(BATCH_SIZE, 2048)
    mock_create_model.return_value = mock_image_encoder

    model = MediLLMModel(mode="multimodal")
    model.eval()
    outputs = model(
        input_ids=dummy_inputs["input_ids"],
        atttention_mask=dummy_inputs["attention_mask"],
        image=dummy_inputs["image"],
    )

    assert outputs.shape == (BATCH_SIZE, 3)
    probs = torch.softmax(outputs, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-5)
