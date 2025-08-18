import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from app.utils.inference_utils import load_model
from app.utils.attention_utils import extract_token_attention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Load model from config
model = load_model("multimodal", "medi_llm_state_dict_multimodal.pth")

# Test input
text = "Patient-A reports shortness of breath and low oxygen levels."
tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
input_ids = tokens["input_ids"].to(DEVICE)
mask = tokens["attention_mask"].to(DEVICE)

# Extract token attention
attention = extract_token_attention(model, tokenizer, input_ids, mask)
print(attention)
