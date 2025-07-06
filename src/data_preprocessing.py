import os
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer

# Disable parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# This script preprocesses EMR data and images for a clinical NLP task.
# It loads a CSV file containing EMR records, tokenizes the text using a
# clinical BERT tokenizer, and preprocesses images for further analysis.
# Import necessary libraries

# Use a clinical tokenizer ( or basic BERT )
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


def load_data(csv_path):
    """
    Load data from a CSV file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def preprocess_text(text):
    """
    Preprocess text data.
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    return tokenizer(
        text, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
    )


def preprocess_image(image_path, image_size=(224, 224)):
    """
    Preprocess image data.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")

    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size)
    return img


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "emr_records.csv")
    df = load_data(data_path)
    print("Data loaded successfully.")
    # apply function applies to each row in the 'image_path' column and joins
    # the base directory with the relative path
    df["image_path"] = df["image_path"].apply(lambda p: os.path.join(base_dir, p))
    print("Sample record:")
    print(df.iloc[0])

    text_encoding = preprocess_text(df.iloc[0]["emr_text"])
    print("Tokenized EMR:")
    print(text_encoding.input_ids.shape)

    img = preprocess_image(df.iloc[0]["image_path"])
    img.show()  # Display the image
