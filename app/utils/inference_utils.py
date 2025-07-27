import sys
import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.multimodal_model import MediLLMModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map
inv_map = {0: "low", 1: "medium", 2: "high"}

# Tokenizer and image transform
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


def load_model(mode, model_path, config_path=str(Path("config/config.yaml").resolve())):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)[mode]

    model = MediLLMModel(
        mode=mode,
        dropout=config["dropout"],
        hidden_dim=config["hidden_dim"]
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict(model, mode, emr_text=None, image=None):
    with torch.no_grad():
        input_ids = attention_mask = img_tensor = None

        if mode in ["text", "multimodal"] and emr_text:
            text_tokens = tokenizer(
                emr_text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            input_ids = text_tokens["input_ids"].to(DEVICE)
            attention_mask = text_tokens["attention_mask"].to(DEVICE)

        if mode in ["image", "multimodal"] and image:
            img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

        output = model(input_ids=input_ids, attention_mask=attention_mask, image=img_tensor)
        pred = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).squeeze()[pred].item()

        return f"{inv_map[pred]} (Confidence: {confidence:.2f})"
