import os
import sys
import torch 
import optuna
import yaml
import json
import wandb 
import argparse
import matplotlib.pyplot as plt 
import seaborn as sns 

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset 
from torch.nn import CrossEntropyLoss
from torch.optim import Adam 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# Setup base path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Automatically add Project root to python import path
base_dir = os.path.dirname(os.path.dirname(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)


from src.triage_dataset import TriageDataset
from src.multimodal_model import MediLLMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stratified_split(dataset, val_ratio=0.2, seed=42, label_column="triage_level"):
    label_map = {"low": 0, "medium": 1, "high": 2}
    labels = [dataset.df.iloc[i][label_column] for i in range(len(dataset))]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(range(len(dataset)), labels))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def objective(trial, mode):
    wandb.init(
        project=f"mediLLM-tune-{mode}",
        name=f"{mode}-trial-{trial.number}-v5-{wandb.util.generate_id()}",
        group="SoftLabelTrials",
        config={
            "dataset_version": "softlabels",
            "dataset_size": 900,
            "mode": mode
        }
    )

    # --- Hyperparameters ---
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    batch_size = trial.suggest_categorical("bs", [4, 8, 16])

    model = MediLLMModel(dropout=dropout, hidden_dim=hidden_dim, mode=mode).to(device)
    wandb.watch(model)

    dataset = TriageDataset(os.path.join(base_dir, "data", "emr_records.csv"))
    train_set, val_set = stratified_split(dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(2):
        model.train()
        loop = tqdm(train_loader, desc=f"[{mode}] Epoch {epoch+1}/2", leave=False)
        for batch in loop:
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
        for batch in tqdm(val_loader, desc=f"[{mode}] Validating", leave=False):
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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="weighted")
    acc = accuracy_score(all_labels, all_preds)

    # Log to W&B and Optuna
    wandb.log({
        "val_f1_score": f1,
        "val_accuracy": acc,
        "lr": lr,
        "dropout": dropout,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size
    })

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["low", "medium", "high"],
                yticklabels=["low", "medium", "high"])
    plt.title(f"Confusion Matrix - {mode} Trial {trial.number}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    wandb.log({f"{mode}_confusion_matrix/trial_{trial.number}": wandb.Image(plt)})
    plt.close()
    
    return f1

def get_args():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials to run")
    parser.add_argument("--mode", type=str, choices=["text", "image", "multimodal"], required=True, help="Input mode")
    return parser.parse_args()

if __name__=="__main__":
    args = get_args()
    mode = args.mode 

    study = optuna.create_study(
        study_name=f"mediLLM_{mode}_optuna",
        direction="maximize"
    )
    with tqdm(total=args.n_trials, desc=f"Optuna Trials [{mode}]") as pbar:
        def wrapped_objective(trial):
            try:
                return objective(trial, mode)
            finally:
                wandb.finish()
                pbar.update(1)
    
        study.optimize(wrapped_objective, n_trials=args.n_trials)

    print(f"✅ Best F1 score for {mode}: {study.best_value}")
    print(f"✅ Best hyperparameters: {study.best_params}")

    # Save best hyperparameters to JSON per mode
    json_path = os.path.join(base_dir, "assets", "best_hyperparams.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    best_params_entry = {
        "lr": float(study.best_params["lr"]),
        "dropout": float(study.best_params["dropout"]),
        "hidden_dim": int(study.best_params["hidden_dim"]),
        "batch_size": int(study.best_params["bs"]),
        "epochs": 5
    }

    # Load existing or start new
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            best_params_all = json.load(f)

    else:
        best_params_all = {}

    best_params_all[mode] = best_params_entry

    # Write back
    with open(json_path, "w") as f:
        json.dump(best_params_all, f, indent=4)

    print(f"✅ Saved best hyperparameters for [{mode}] to best_hyperparams.json")

    # Export to config.yaml
    config_path = os.path.join(base_dir, "config", "config.yaml")
    
    # Make sure config directory exists in the root
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

    config[mode] = {
        "lr": float(study.best_params["lr"]),
        "dropout": float(study.best_params["dropout"]),
        "hidden_dim": int(study.best_params["hidden_dim"]),
        "batch_size": int(study.best_params["bs"]),
        "epochs": 5
    }

    # Export to config.yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"✅ Best hyperparameters for [{mode}] saved in config.yaml")
    

    