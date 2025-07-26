import torch  # PyTorch core utility for model training
import os
import sys
import yaml
import json
import argparse
import matplotlib.pyplot as plt  # for plotting
import random
import numpy as np

from tqdm import tqdm  # loading bar for loops
from torch.utils.data import (
    DataLoader,
    Subset,
)  # Dataloader to batch and feed data to model,

# random split to split dataset into train and validation sets
from torch.nn import (
    CrossEntropyLoss,
)  # PyTorch core utility for model training
from torch.optim import Adam  # PyTorch core utility for model training,

# Adam is the Optimizer, a gradient descent model
from sklearn.metrics import accuracy_score, f1_score, classification_report  # Evaluation metrics
from sklearn.model_selection import StratifiedShuffleSplit

# Setup base path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Automatically add Project root to python import path
base_dir = os.path.dirname(os.path.dirname(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.triage_dataset import TriageDataset  # Dataset Class
from src.multimodal_model import MediLLMModel  # Mutlimodal Model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(mode):
    config_path = os.path.join(base_dir, "config", "config.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # If the config file doesn't exist, create it defaults for all modes
    if not os.path.exists(config_path):
        default_config = {
            "text": {
                "lr": 2e-5,
                "dropout": 0.3,
                "hidden_dim": 256,
                "batch_size": 8,
                "epochs": 5,
            },
            "image": {
                "lr": 2e-5,
                "dropout": 0.3,
                "hidden_dim": 256,
                "batch_size": 8,
                "epochs": 5,
            },
            "multimodal": {
                "lr": 2e-5,
                "dropout": 0.3,
                "hidden_dim": 256,
                "batch_size": 8,
                "epochs": 5,
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(default_config, f)
    # otherwise export to yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if mode not in config:
        raise ValueError(f"No config found for mode '{mode}' in config.yaml")
    return config[mode]


def stratified_split(dataset, val_ratio=0.2, seed=42):
    labels = [dataset.df.iloc[i]["triage_level"] for i in range(len(dataset))]
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=seed,
    )
    tran_idx, val_idx = next(sss.split(range(len(dataset)), labels))
    return Subset(dataset, tran_idx), Subset(dataset, val_idx)


# Function to instantiate model and data, train, validate, plot results
# and save the model
def train_model(mode="multimodal", use_wandb=False):
    set_seed(42)

    if use_wandb:
        import wandb

    cfg = load_config(mode)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Use GPU if available or else use CPU
    dataset_dir = os.path.join(base_dir, "data", "emr_records.csv")
    dataset_kwargs = {"csv_file": dataset_dir, "mode": mode}
    if mode in ["image", "multimodal"]:
        image_dir = os.path.join(base_dir, "data", "images")
        dataset_kwargs["image_base_dir"] = image_dir

    dataset = TriageDataset(**dataset_kwargs)
    model = MediLLMModel(
        dropout=cfg["dropout"], hidden_dim=cfg["hidden_dim"], mode=mode
    ).to(
        device
    )  # moves the model to selected device

    if use_wandb:
        # Initialize Weights & Biases
        wandb.init(
            project="MediLLM_Final_v2",
            name=f"train_{mode}",
            config=cfg
        )
        wandb.config.update({"mode": mode})

    train_set, val_set = stratified_split(dataset)
    batch_size = cfg["batch_size"]

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )  # Create data in batches to the model
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Calculate difference between model prediction and true labels
    criterion = CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(), lr=cfg["lr"]
    )  # Adaptive learning rate optimizer for fast-converging

    # Lists to store accuracy per epoch for plotting
    train_acc, val_acc = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(cfg["epochs"]):
        model.train()  # Activate training the model, enable dropout
        all_preds, all_labels = [], []

        for batch in tqdm(
            train_loader, desc=f"[{mode}] Epoch {epoch + 1}"
        ):  # Load a batch of text, images, and labels to GPU or CPU
            input_ids = batch.get("input_ids", None)
            attention_mask = batch.get("attention_mask", None)
            images = batch.get("image", None)
            labels = batch["label"].to(device)

            if input_ids is not None:
                input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            if images is not None:
                images = images.to(device)
            """
                Each batch looks like this
                {
                    "input_ids":       torch.Size([8, 128]),
                    "attention_mask":  torch.Size([8, 128]),
                    "image":           torch.Size([8, 3, 224, 224]),
                    "label":           torch.Size([8])
                }
            """
            optimizer.zero_grad()  # Zero out gradients from previous batch
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=images,
            )  # Forward pass through the model
            loss = criterion(outputs, labels)  # Compute loss value
            loss.backward()  # Back propagation to compute gradients
            optimizer.step()  # Adjust the weights using gradients

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            # Get the predicted class per sample and convert to CPU & Numpy for
            # easier comparison
            all_preds.extend(preds)
            # Save predictions for metric computation.
            # extend() appends each element of preds to the list
            all_labels.extend(
                labels.cpu().numpy()
            )  # Save labels for metric computation

        # Calculating classification metrics (Accuracy and F1)
        acc = accuracy_score(all_labels, all_preds)  # Evaluate full-epoch performance
        f1 = f1_score(all_labels, all_preds, average="weighted")
        # 1) binary: Binary Classification(F1 score of +ve class only)
        # 2) macro: Computes F1 for each class independently, then averages,
        #    treats all classes equally
        # 3) micro: Flattens all true and predicted labels and then computes
        #    global TP, FP, FN and gets F1 from that, works well with
        #    imbalanced data, equal to accuracy in binary classification and
        #    different in multi-class/multi-label
        # 4) weighted: calculates F1 for each class, then averages them using
        #    number of samples, avoids bias, real-world and imbalanced classes,
        #    per-class performance
        # 5) samples: used for multi-label classification, computes F1 for each
        #    instance, then averages across all samples, row-wise,
        #    not class-wise
        train_acc.append(acc)  # Append to a list for plotting
        train_f1s.append(f1)

        print(f"Train Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

        # Validation loop
        model.eval()  # Deactivates dropnot and batchnorm for inference
        val_preds, val_labels = [], []

        with torch.no_grad():  # Disables autograd to save memory
            for batch in val_loader:
                # Load batch of validation data text, images, labels
                # to GPU or CPU
                input_ids = batch.get("input_ids", None)
                attention_mask = batch.get("attention_mask", None)
                images = batch.get("image", None)
                labels = batch["label"].to(device)

                if input_ids is not None:
                    input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                if images is not None:
                    images = images.to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image=images,
                )
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        val_acc_epoch = accuracy_score(
            val_labels, val_preds
        )  # Validation metrics, accuracy: how many items did your model get
        # right out of total items
        val_f1 = f1_score(val_labels, val_preds, average="weighted")
        # F1 score weights in both precision and recall, uses harmonic mean to
        # punish imbalance. If one of the two is low, it drags the accuracy
        # score down.
        # Precision: How careful is the model when classifying an item
        # (TP / (TP + FP)).
        # Recall: How many real items did it actually spot
        # (TP / (TP + FN)).
        val_acc.append(val_acc_epoch)
        val_f1s.append(val_f1)

        print(f"Val Accuracy: {val_acc_epoch:.4f}, F1 Score: {val_f1:.4f}")

        # Log to Weights & Biases
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/accuracy": acc,
                "train/f1": f1,
                "val/accuracy": val_acc_epoch,
                "val/f1": val_f1
            })

    # Save model
    model_path = os.path.join(base_dir, f"medi_llm_state_dict_{mode}.pth")
    torch.save(
        model.state_dict(), model_path
    )  # Saves only model weights and biases
    print(f"💾 Saved model weights and biases to {model_path}")

    # Save to Weights & Biases
    if use_wandb:
        wandb.save(model_path)

    # Plot accuracy
    plot_path = os.path.join(base_dir, "assets", f"model_training_curve_{mode}.png")
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.legend()
    plt.title(f"Accuracy: Train vs Validation ({mode})")
    plt.savefig(plot_path)
    print(f"✅ Saved training curve to {plot_path}")

    if use_wandb:
        wandb.log({"training_curve": wandb.Image(plot_path)})

    # Save training metrics to JSON
    results = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_f1": train_f1s,
        "val_f1": val_f1s,
        "final_train_acc": train_acc[-1],
        "final_val_acc": val_acc[-1],
        "final_train_f1": train_f1s[-1],
        "final_val_f1": val_f1s[-1]
    }

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"metrics_{mode}.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📊 Saved training metrics to {results_path}")

    # Classification Report
    class_report = classification_report(val_labels, val_preds, output_dict=True, zero_division=0, target_names=["low", "medium", "high"])
    print("\n🗓️ Classification Report (Per Class on Validation Set):")
    for cls, metrics in class_report.items():
        if cls in ["low", "medium", "high"]:
            print(f"{cls:>9} -> Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}")

    class_report_path = os.path.join(results_dir, f"classification_report_{mode}.json")
    with open(class_report_path, "w") as f:
        json.dump(class_report, f, indent=2)
    print(f"📊 Saved per-class metrics to {class_report_path}")

    if use_wandb:
        for cls in ["low", "medium", "high"]:
            wandb.log({
                f"classwise/{cls}_precision": class_report[cls]["precision"],
                f"classwise/{cls}_recall": class_report[cls]["recall"],
                f"classwise/{cls}_f1": class_report[cls]["f1-score"],
            })
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["text", "image", "multimodal"], default="multimodal"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        mode=args.mode,
        use_wandb=args.wandb
    )  # Only runs if file is run directly not when it is imported
