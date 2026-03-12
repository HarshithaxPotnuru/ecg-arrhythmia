import numpy as np
import scipy.signal as sg

def bandpass(sig, fs=250, low=0.5, high=40):
    nyq = 0.5 * fs
    b, a = sg.butter(4, [low/nyq, high/nyq], btype='band')
    return sg.filtfilt(b, a, sig, axis=1)

def normalize(sig):
    return (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-8)

def pad_signal(x, max_len):
    mask = np.zeros(max_len)
    out = np.zeros((x.shape[0], max_len))
    L = min(max_len, x.shape[1])
    out[:, :L] = x[:, :L]
    mask[:L] = 1
    return out, mask
