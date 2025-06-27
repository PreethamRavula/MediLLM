import torch # PyTorch core utility for model training 
import os
from torch.utils.data import DataLoader, Subset # Dataloader to batch and feed data to model, random split to split dataset into train and validation sets
from torch.nn import CrossEntropyLoss # PyTorch core utility for model training
from torch.optim import Adam # PyTorch core utility for model training, Adam is the Optimizer a gradient descent model
from sklearn.metrics import accuracy_score, f1_score # Evaluation metrics
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm # loading bar for loops
import matplotlib.pyplot as plt # for plotting
from src.triage_dataset import TriageDataset # Dataset Class
from src.multimodal_model import MediLLMModel # Mutlimodal Model

base_dir = os.path.dirname(os.path.dirname(__file__)) # Project directory

def load_config():
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
    
    # otherwise export to yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def stratified_split(dataset, val_ratio=0.2, seed=42):
    labels = [dataset.df.iloc[i]["triage_level"] for i in range(len(dataset))]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tran_idx, val_idx = next(sss.split(range(len(dataset)), labels))
    return Subset(dataset, tran_idx), Subset(dataset, val_idx)

def train_model(): # Function to instantiate model and data, train, validate, plot results and save the model
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available or else use CPU
    
    dataset_dir = os.path.join(base_dir, "data", "emr_records.csv")
    dataset = TriageDataset(
        csv_file=dataset_dir
    )

    model = MediLLMModel(
        dropout=config["model"]["dropout"],
        hidden_dim=config["model"]["hidden_dim"]
    ).to(device) # moves the model to selected device
 
    train_set, val_set = stratified_split(dataset)

    train_loader = DataLoader(train_set, batch_size=config["train"]["batch_size"], shuffle=True) # Create data in batches to the model
    val_loader = DataLoader(val_set, batch_size=config["train"]["batch_size"])

    criterion = CrossEntropyLoss() # Calculate difference between model prediction and true labels
    optimizer = Adam(model.parameters(), lr=config["train"]["lr"]) # Adaptive learning rate optimizer for fast-converging

    train_acc, val_acc = [], [] # Lists to store accuracy per epoch for plotting

    for epoch in range(config["train"]["epochs"]):
        model.train() # Activate training the model, enable dropout
        all_preds, all_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"): # Load a batch of text, images, and labels to GPU or CPU
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            """
                Each batch looks like this
                {
                    "input_ids":       torch.Size([8, 128]),
                    "attention_mask":  torch.Size([8, 128]),
                    "image":           torch.Size([8, 3, 224, 224]),
                    "label":           torch.Size([8])
                }
            """
            optimizer.zero_grad() # Zero out gradients from previous batch
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, image=images) # Forward pass through the model
            loss = criterion(outputs, labels) # Compute loss value
            loss.backward() # Back propagation to compute gradients
            optimizer.step() # Adjust the weights using gradients

            preds = torch.argmax(outputs, dim=1).cpu().numpy() # Get the predicted class per sample and covert to CPU & Numpy for easier comparision
            all_preds.extend(preds) # Save predictions for metric computation, extend() appends each element of preds to the list
            all_labels.extend(labels.cpu().numpy()) # Save labels for metric computation
        
        # Calculating classification metrics (Accuracy and F1)
        acc = accuracy_score(all_labels, all_preds) # Evaluate full-epoch performance
        f1 = f1_score(all_labels, all_preds, average="weighted") # 1) binary: Binary Classification(F1 score of +ve class only); 2) macro: Computes F1 for each class independently, then averages, treats all classes equally; 3) micro: Flattens all true and predicted labels and the computes global TP, FP, FN and gets F1 from that, works well with imbalanced data, equal to accuracy in binary classification and different in multi-class/multi-label; 4) Weighted: calculates F1 for each class, then averages them using number of samples, avoids bias, real-world and imbalanced classes, per-class performance; 5) samples: used for multi-label classification, computes F1 for each instance, then averages across all samples, row-wise, not class-wise
        train_acc.append(acc) # Append to a list for plotting

        print(f"Train Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

        # Validation loop
        model.eval() # Deactivates dropnot and batchnorm for inference
        val_preds, val_labels = [], []

        with torch.no_grad(): # Disables autograd to save memory
            for batch in val_loader: # Load batch of validation data text, images, labels to GPU or CPU
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids, attn_mask, images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        val_acc_epoch = accuracy_score(val_labels, val_preds) # Validation metrics, accuracy: how mant items did your model get right out of total items
        val_f1 = f1_score(val_labels, val_preds, average="weighted") # F1 score weights in both precision and recall, uses hoarmonic mean to punish imbalance, if one of the 2 is low it drags the accuracy score down, precision: How careful is the model when classfying an item (TP / (TP + FP)), Recall: Hom many real items did it actually spot (TP / (TP + FN))
        val_acc.append(val_acc_epoch)

        print(f"Val Accuracy: {val_acc_epoch:.4f}, F1 Score: {val_f1:.4f}")

    # Save model
    model_path = os.path.join(base_dir, "medi_llm_model.pth")
    torch.save(model.state_dict(), model_path) # Saves the model weights only not total architecture to reuse later

    # Plot accuracy
    save_dir = os.path.join(base_dir, "assets", "model_training_curve.png")
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    plt.savefig(save_dir)

if __name__=="__main__":
    train_model() # Only runs if file is run directly not when it is imported



