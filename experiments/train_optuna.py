import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Automatically add Project root to python import path
base_dir = os.path.dirname(os.path.dirname(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

import torch 
import optuna 
from torch.utils.data import DataLoader, Subset 
from torch.nn import CrossEntropyLoss
from torch.optim import Adam 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt 
import seaborn as sns 
import wandb 
import json
import yaml
import argparse

from src.triage_dataset import TriageDataset
from src.multimodal_model import MediLLMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stratified_split(dataset, val_ratio=0.2, seed=42, label_column="triage_level"):
    label_map = {"low": 0, "medium": 1, "high": 2}
    labels = [dataset.df.iloc[i][label_column] for i in range(len(dataset))]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(range(len(dataset)), labels))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def objective(trial):
    wandb.init(
        project="mediLLM-v2",
        name=f"trial-{trial.number}-v4-{wandb.util.generate_id()}",
        group="SoftLabelTrials",
        config={
            "dataset_version": "softlabels",
            "dataset_size": 900
        }
    )

    # --- Hyperparameters ---
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    batch_size = trial.suggest_categorical("bs", [4, 8, 16])

    model = MediLLMModel(dropout=dropout, hidden_dim=hidden_dim).to(device)
    wandb.watch(model)

    dataset = TriageDataset(os.path.join(base_dir, "data", "emr_records.csv"))
    train_set, val_set = stratified_split(dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(2):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/2", leave=False)
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"\n[Trial {trial.number}] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["low", "medium", "high"]))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["low", "medium", "high"],
                yticklabels=["low", "medium", "high"])
    plt.title(f"Confusion Matrix - Trial {trial.number}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    wandb.log({f"confusion_matrix/trial_{trial.number}": wandb.Image(plt)})
    plt.close()
    
    # Log to W&B and Optuna
    wandb.log({
        "f1_score": f1,
        "accuracy": accuracy_score(all_labels, all_preds),
        "lr": lr,
        "dropout": dropout,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size
    })
    return f1

def get_args():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials to run")
    return parser.parse_args()

if __name__=="__main__":
    args = get_args()

    study = optuna.create_study(
        study_name="mediLLM_v2",
        direction="maximize"
    )
    with tqdm(total=args.n_trials, desc="Optuna Trials") as pbar:
        def wrapped_objective(trial):
            try:
                result = objective(trial)
                return result
            finally:
                wandb.finish()
                pbar.update(1)
    
        study.optimize(wrapped_objective, n_trials=args.n_trials)

    print("Best F1 score achieved:", study.best_value)
    print("Best hyperparameters:", study.best_params)

    # Save as JSON
    assets_dir = os.path.join(base_dir, "assets")

    # Make sure assets directory exists in the root
    os.makedirs(assets_dir, exist_ok=True)

    # Save the best hyperparameters
    with open(os.path.join(assets_dir, "best_hyperparams.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)

    # Export to config.yaml
    config_dir = os.path.join(base_dir, "config")
    config_path = os.path.join(config_dir, "config.yaml")
    
    # Make sure config directory exists in the root
    os.makedirs(config_dir, exist_ok=True)

    # If the config file doesn't exist, create a default one
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write(
                "model:\n"
                "  dropout: 0.3\n"
                "  hidden_dim: 256\n\n"
                "train:\n"
                "  lr: 2e-5\n"
                "  batch_size: 8\n"
                "  epochs: 5\n\n"
                "wandb:\n"
                " project: medi-llm-final\n"
            )

    # Export to config.yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["dropout"] = float(study.best_params["dropout"])
    cfg["model"]["hidden_dim"] = int(study.best_params["hidden_dim"])
    cfg["train"]["lr"] = float(study.best_params["lr"])
    cfg["train"]["batch_size"] = int(study.best_params["bs"])

    # Save updated config
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    

    