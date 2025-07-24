import argparse
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from src.triage_dataset import TriageDataset
from src.multimodal_model import MediLLMModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path="config/config.yaml", mode="multimodal"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config[mode]


def predict(model, dataloader, device):
    model.eval()
    all_preds, all_truths = [], []
    all_texts, all_paths, all_ids = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            input_ids = batch.get("input_ids", None)
            attention_mask = batch.get("attention_mask", None)
            images = batch.get("image", None)

            if input_ids is not None:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
            if images is not None:
                images = images.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)

            all_truths.extend(batch.get("triage_level", [-1] * len(preds)))
            all_texts.extend(batch.get("emr_text", [""] * len(preds)))
            all_paths.extend([str(p) for p in batch.get("image_path", [""] * len(preds))])
            all_ids.extend(batch.get("patient_id", [f"patient_{i}" for i in range(len(preds))]))

    return all_preds, all_truths, all_texts, all_paths, all_ids


def inverse_label_map():
    return {0: "low", 1: "medium", 2: "high"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="test_samples.csv", help="Path to test file")
    parser.add_argument("--mode", type=str, choices=["text", "image", "multimodal"], default="multimodal", help="mode of data")
    parser.add_argument("--model_path", type=str, required=True, help="path to the model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--image_dir", type=str, default="data/images", help="path to images folder")
    parser.add_argument("--output_csv", type=str, help="Optional custome path to output file")
    parser.add_argument("--batch_size", type=int, help="Optional override for batch size")
    parser.add_argument("--save_misclassified_only", action="store_true", help="Save only misclassified samples")
    args = parser.parse_args()

    # Checks
    if not Path(args.csv_path).exists():
        raise FileNotFoundError(f"❌ CSV file not found at {args.csv_path}")

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"❌ Model checkpoint not found at: {args.model_path}")

    if not Path(args.config).exists():
        raise FileNotFoundError(f"❌ Config file not found at: {args.config}")

    if args.mode in ["image", "multimodal"] and not Path(args.image_dir).exists():
        raise FileNotFoundError(f"❌ Image directory not found at: {args.image_dir}")

    # Always generate mode-specific output file if not provided
    if not args.output_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = f"predictions_{args.mode}_{timestamp}.csv"

    config = load_config(config_path=args.config, mode=args.mode)
    batch_size = args.batch_size or config["batch_size"]

    dataset = TriageDataset(
        csv_file=args.csv_path,
        mode=args.mode,
        image_base_dir=args.image_dir
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = MediLLMModel(
        mode=args.mode,
        dropout=config["dropout"],
        hidden_dim=config["hidden_dim"]
    )
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE)

    preds, truths, texts, paths, ids = predict(model, dataloader, DEVICE)

    label_inv_map = inverse_label_map()
    pred_labels = [label_inv_map(p) for p in preds]
    true_labels = [label_inv_map(t) if t in label_inv_map else "" for t in truths]

    df = pd.DataFrame({
        "patient_id": ids,
        "predicted": pred_labels,
        "true": true_labels,
        "emr_text": texts,
        "image_path": paths,
    })

    # Filter misclassified rows if needed
    if args.save_missclassified_only:
        df = df[df["predicted"] != df["true"]]

    print(df[["patient_id", "predicted", "true"]])
    # Ensure output directory exists
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save predictions
    df.to_csv(args.output_csv, index=False)
    print(f"✅ Saved predictions to {args.output_csv}")
    print(f"\n🔎 Processed {len(preds)} samples ({'missclassified only' if args.save_misclassified_only else 'all'}).")

    # print classification report + metrics if labels exist
    if all(label in ["low", "medium", "high"] for label in true_labels):
        print("\n📊 Classification Report:")
        print(classification_report(true_labels, pred_labels))

        acc = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average="weighted")
        print(f"\n🎯 Accuracy: {acc:.4f}")
        print(f"🎯 Weighted F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
