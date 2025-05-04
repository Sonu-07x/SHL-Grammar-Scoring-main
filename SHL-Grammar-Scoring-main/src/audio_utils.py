import librosa
import numpy as np

def extract_features(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)
