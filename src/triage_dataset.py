import os
import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pandas as pd
from transformers import AutoTokenizer

# Check if running in CI environment
IS_CI = os.getenv("CI", "false").lower() == "true"


class TriageDataset(Dataset):
    def __init__(
        self,
        csv_file,
        tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
        max_length=128,
        transform=None,
        mode="multimodal",
        image_base_dir=None,
    ):
        assert mode in [
            "text",
            "image",
            "multimodal",
        ], "Mode must be one of: 'text', 'image', or 'multimodal'"

        self.df = pd.read_csv(csv_file)  # Create a dataframe from csv file
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.mode = mode.lower()
        if self.mode in ["image", "multimodal"]:
            if image_base_dir is None:
                raise ValueError("image directory must be provided for image or multimodal mode.")
            self.image_base_dir = Path(image_base_dir).resolve()

        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    transforms.Resize((256, 256)),  # Resize first
                    transforms.RandomResizedCrop(
                        224,
                        scale=(0.9, 1.0),
                        interpolation=InterpolationMode.BILINEAR,
                    ),  # Slight zoom-in/out
                    transforms.RandomRotation(degrees=10),  # + or - 10° rotation
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.3
                    ),  # simulate slight imaging variations
                    transforms.GaussianBlur(kernel_size=3),
                    transforms.ToTensor(),
                ]
            )
        )

        # Label mapping
        self.label_map = {"low": 0, "medium": 1, "high": 2}

    def __len__(self):
        return len(
            self.df
        )  # returns number of rows so dataloader can know how many batches
        # to prepare

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        output = {}

        if self.mode in ["text", "multimodal"]:
            # Process text
            text = row["emr_text"]
            tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            # removing batch dimension from tokenized tensors
            output["input_ids"] = tokens["input_ids"].squeeze(0)
            output["attention_mask"] = tokens["attention_mask"].squeeze(0)

            # for inference
            if "text" in self.df.columns:
                output["raw_text"] = text

        if self.mode in ["image", "multimodal"]:
            # Process image
            image_path = Path(row["image_path"])

            if not image_path.is_absolute():
                if image_path.parts[:2] == ("data", self.image_base_dir.name):
                    image_path = self.image_base_dir.parent / Path(*image_path.parts[1:])
                else:
                    image_path = self.image_base_dir / image_path

            if not image_path.exists():
                msg = f"Image file not found: {image_path}"
                raise FileNotFoundError(f"[CI] {msg}" if IS_CI else f"[LOCAL] {msg}")

            image = Image.open(image_path).convert("RGB")
            output["image"] = self.transform(image)

        # Label
        if "triage_level" in row and row["triage_level"] in self.label_map:
            output["label"] = torch.tensor(
                self.label_map[row["triage_level"]], dtype=torch.long
            )

        # fields for inference output
        if "patient_id" in row:
            output["patient_id"] = row["patient_id"]

        if "emr_text" in row and "emr_text" not in output:
            output["emr_text"] = row["emr_text"]

        if "image_path" in row and "image_path" not in output:
            output["image_path"] = str(image_path) if "image_path" in locals() else row["image_path"]

        return output
