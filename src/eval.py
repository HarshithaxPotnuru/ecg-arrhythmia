import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss

from src.dataloader import ECGDataset
from src.models import CNN1D, CNN_BiLSTM, ECGTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="cnn")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    dataset = ECGDataset("data/records.csv")
    loader = DataLoader(dataset, batch_size=args.batch)

    # Model selection
    models = {
        "cnn": CNN1D(9),
        "cnn_bilstm": CNN_BiLSTM(9),
        "transformer": ECGTransformer(9),
    }

    model = models[args.model].to(device)

    print("Loading checkpoint:", args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    preds = []
    trues = []

    print("Running evaluation...")

    with torch.no_grad():
        for x, mask, y in loader:
            x = x.to(device)
            outputs = torch.sigmoid(model(x, mask)).cpu().numpy()

            preds.append(outputs)
            trues.append(y.numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)

    # Binary predictions
    threshold = 0.3
    preds_bin = (preds > threshold).astype(int)
    print("Using threshold:", threshold)


    # Metrics
    macro_f1 = f1_score(trues, preds_bin, average="macro")
    hamming = hamming_loss(trues, preds_bin)

    try:
        auc = roc_auc_score(trues, preds, average="macro")
    except:
        auc = "Could not compute"

    print("\n===== Evaluation Results =====")
    print("Macro F1:", macro_f1)
    print("Hamming Loss:", hamming)
    print("AUROC:", auc)


if __name__ == "__main__":
    main()
