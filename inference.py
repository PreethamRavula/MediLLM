import argparse
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms

from src.triage_dataset import TriageDataset
from src.multimodal_model import MediLLMModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


def load_config(config_path="config/config.yaml", mode="multimodal"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config[mode]


def predict(model, dataloader, device):
    model.eval()
    all_preds, all_truths = [], []
    all_texts, all_paths, all_ids = [], [], []

    inv_map = {0: "low", 1: "medium", 2: "high"}

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

            batch_size = len(preds)

            # Patient ID (original from dataset)
            all_ids.extend(batch["patient_id"])

            # EMR Text
            all_texts.extend(batch.get("emr_text", [""] * batch_size))

            # Image Path
            all_paths.extend(batch.get("image_path", [""] * batch_size))

            # True Labels
            if "label" in batch:
                all_truths.extend([inv_map.get(label.item(), "") for label in batch["label"]])
            else:
                all_truths.extend([""] * batch_size)

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
        image_base_dir=args.image_dir,
        transform=inference_transform  # avoid random augmentations during inference
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
    pred_labels = [label_inv_map[p] for p in preds]

    df = pd.DataFrame({
        "patient_id": ids,
        "predicted": pred_labels,
        "truth_label": truths,
        "emr_text": texts,
        "image_path": paths,
    })

    # Filter misclassified rows if needed
    if args.save_misclassified_only:
        df = df[df["predicted"] != df["truth_label"]]

    print(df[["patient_id", "predicted", "truth_label"]])
    # Ensure output directory exists
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save predictions
    df.to_csv(args.output_csv, index=False)
    print(f"✅ Saved predictions to {output_path}")
    print(f"\n🔎 Processed {len(preds)} samples ({'missclassified only' if args.save_misclassified_only else 'all'}).")

    # print classification report + metrics if labels exist
    if all(label in ["low", "medium", "high"] for label in truths):
        print("\n📊 Classification Report:")
        print(classification_report(truths, pred_labels))

        acc = accuracy_score(truths, pred_labels)
        f1 = f1_score(truths, pred_labels, average="weighted")
        print(f"\n🎯 Accuracy: {acc:.4f}")
        print(f"\n🎯 Weighted F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
