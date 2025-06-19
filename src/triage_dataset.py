import os
import torch 
from torch.utils.data import Dataset 
from PIL import Image 
from torchvision import transforms 
import pandas as pd 
from transformers import AutoTokenizer

class TriageDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name="emilyalsentzer/Bio_ClinicalBERT", max_length=128, transform=None):
        self.df = pd.read_csv(csv_file) # Create a dataframe from csv file
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Label mapping 
        self.label_map = {"low": 0, "medium": 1, "high": 2}

    def __len__(self):
        return len(self.df) # returns number of rows so dataloader can know how many batches to prepare
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx] 

        # Process text
        text = row["emr_text"]
        tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        # Process image
        base_dir = os.path.dirname(os.path.dirname(__file__))
        image_path = os.path.join(base_dir, row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Label 
        label = self.label_map[row["triage_level"]]
        return { # removing batch dimension from tokenized tensors
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "image": image,
            "label": torch.tensor(label, dtype=torch.long) # Loss functions like crossentropy expects label as LongTensor(int64), not floats or other types
        }
