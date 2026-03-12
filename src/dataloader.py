import wfdb
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from src.preprocess import bandpass, normalize, pad_signal

class ECGDataset(Dataset):
    def __init__(self, csv_file, max_len=15000):
        df = pd.read_csv(csv_file)

        self.records = df["record"].tolist()
        raw_labels = df["labels"].astype(str).str.split("|").tolist()

        self.mlb = MultiLabelBinarizer()
        self.Y = self.mlb.fit_transform(raw_labels)

        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        sig, _ = wfdb.rdsamp(self.records[idx])
        sig = sig.T
        sig = bandpass(sig)
        sig = normalize(sig)
        sig, mask = pad_signal(sig, self.max_len)

        return (
            torch.tensor(sig, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(self.Y[idx], dtype=torch.float32),
        )
