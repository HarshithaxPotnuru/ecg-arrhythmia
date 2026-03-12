import torch
import torch.nn as nn

# ---------- 1D CNN ----------
class CNN1D(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x, mask=None):
        x = self.conv(x)
        return self.fc(x.squeeze(-1))


# ---------- Attention ----------
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, 1)

    def forward(self, x):
        w = torch.softmax(self.W(x), dim=1)
        return (w * x).sum(1)


# ---------- CNN + BiLSTM ----------
class CNN_BiLSTM(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.cnn = nn.Conv1d(12, 64, kernel_size=7, padding=3)
        self.lstm = nn.LSTM(64, 128, bidirectional=True, batch_first=True)
        self.attn = Attention(256)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x, mask=None):
        x = self.cnn(x).transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.attn(x)
        return self.fc(x)


# ---------- Transformer ----------
class ECGTransformer(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.embed = nn.Conv1d(12, 128, kernel_size=1)
        enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.trans = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x, mask=None):
        x = self.embed(x).permute(2, 0, 1)  # T,B,C
        x = self.trans(x)
        return self.fc(x.mean(0))
