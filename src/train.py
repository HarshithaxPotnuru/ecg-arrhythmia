import argparse
import os
import re
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.dataloader import ECGDataset
from src.models import CNN1D, CNN_BiLSTM, ECGTransformer
from src.utils import FocalLoss


def get_latest_checkpoint(folder):
    """Find latest epoch checkpoint"""
    if not os.path.exists(folder):
        return None, 0

    files = os.listdir(folder)
    epochs = []

    for f in files:
        match = re.match(r"epoch(\d+)\.pt", f)
        if match:
            epochs.append(int(match.group(1)))

    if len(epochs) == 0:
        return None, 0

    last_epoch = max(epochs)
    path = os.path.join(folder, f"epoch{last_epoch}.pt")
    return path, last_epoch + 1


def compute_class_weights(dataset, device):
    """
    Compute inverse-frequency class weights
    Improves Macro-F1 for imbalanced data
    """
    print("Computing class weights...")

    labels = dataset.Y  # shape: (N, 9)
    class_counts = labels.sum(axis=0)

    # Inverse frequency
    weights = 1.0 / (class_counts + 1e-6)

    # Normalize (important for stability)
    weights = weights / weights.mean()

    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    print("Class counts:", class_counts)
    print("Class weights:", weights)

    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset
    print("Loading dataset...")
    dataset = ECGDataset("data/records.csv")
    print("Total samples:", len(dataset))

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    # Models
    models = {
        "cnn": CNN1D(9),
        "cnn_bilstm": CNN_BiLSTM(9),
        "transformer": ECGTransformer(9),
    }

    model = models[args.model].to(device)
    print("Model:", args.model)

    # ---- Class imbalance handling ----
    class_weights = compute_class_weights(dataset, device)

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = FocalLoss(pos_weight=class_weights)

    # Checkpoints folder
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume if checkpoint exists
    resume_path, start_epoch = get_latest_checkpoint(checkpoint_dir)

    if resume_path is not None:
        print("Resuming from:", resume_path)
        model.load_state_dict(torch.load(resume_path, map_location=device))
    else:
        print("No checkpoint found. Starting fresh.")

    # Training loop
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"\n===== Epoch {epoch} started =====")
        model.train()
        total_loss = 0

        for batch_idx, (x, mask, y) in enumerate(loader):
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(loader)}")

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(x, mask)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} finished. Loss = {avg_loss:.4f}")

        # Save checkpoint
        save_path = os.path.join(checkpoint_dir, f"epoch{epoch}.pt")
        torch.save(model.state_dict(), save_path)
        print("Saved:", save_path)


if __name__ == "__main__":
    main()
