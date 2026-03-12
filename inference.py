import torch
import wfdb
from src.models import CNN1D
from src.preprocess import bandpass, normalize, pad_signal

# Classes
CLASSES = ["AF", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE", "Normal", "Other"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
print("Loading model...")
model = CNN1D(len(CLASSES)).to(device)
model.load_state_dict(torch.load("checkpoints/epoch9.pt", map_location=device))
model.eval()

# Test record (change if needed)
record_path = "data/records/A0001"

print("Reading ECG:", record_path)
signal, _ = wfdb.rdsamp(record_path)

# Preprocess
signal = signal.T
signal = bandpass(signal)
signal = normalize(signal)
signal, _ = pad_signal(signal, 15000)

x = torch.tensor(signal).unsqueeze(0).float().to(device)

# Predict
with torch.no_grad():
    probs = torch.sigmoid(model(x)).cpu().numpy()[0]

print("\nPrediction probabilities:")
for c, p in zip(CLASSES, probs):
    print(f"{c}: {p:.3f}")
