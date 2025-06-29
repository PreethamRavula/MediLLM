import os
import torch 
from torch.utils.data import Dataset 
from PIL import Image 
from torchvision import transforms
from torchvision.transforms import InterpolationMode 
import pandas as pd 
from transformers import AutoTokenizer

class TriageDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name="emilyalsentzer/Bio_ClinicalBERT", max_length=128, transform=None, mode="multimodal"):
        assert mode in ["text", "image", "multimodal"], "Mode must be one of: 'text', 'image', or 'multimodal'"
        
        self.df = pd.read_csv(csv_file) # Create a dataframe from csv file
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.mode = mode.lower()

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)), # Resize first
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=InterpolationMode.BILINEAR), # Slight zoom-in/out
            transforms.RandomRotation(degrees=10), # + or - 10° rotation
            transforms.ColorJitter(brightness=0.3, contrast=0.3), # simulate slight imaging variations
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
        ])

        # Label mapping 
        self.label_map = {"low": 0, "medium": 1, "high": 2}

    def __len__(self):
        return len(self.df) # returns number of rows so dataloader can know how many batches to prepare
    
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
                return_tensors="pt"
            )
            # removing batch dimension from tokenized tensors
            output["input_ids"] = tokens["input_ids"].squeeze(0)
            output["attention_mask"] = tokens["attention_mask"].squeeze(0)

        if self.mode in ["image", "multimodal"]:
            # Process image
            base_dir = os.path.dirname(os.path.dirname(__file__))
            image_path = os.path.join(base_dir, row["image_path"])
            image = Image.open(image_path).convert("RGB")
            output["image"] = self.transform(image)

        # Label 
        output["label"] = torch.tensor(self.label_map[row["triage_level"]], dtype=torch.long)
        
        return output
