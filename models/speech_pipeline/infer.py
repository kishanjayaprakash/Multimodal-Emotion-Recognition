import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from model_utils import SpeechEmotionModel

# ---------------- CONFIG ----------------
# Point this to where train.py saves your best model weights
MODEL_PATH = "../models/speech_model_best.pth"

N_MFCC = 40
INPUT = N_MFCC * 3
HIDDEN = 128
CLASSES = 7

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']


def predict(audio_path):
    """Loads a single audio file, extracts features, and predicts the emotion."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize and load the trained model weights
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Trained model weights not found at {MODEL_PATH}. Run train.py first!")

    model = SpeechEmotionModel(INPUT, HIDDEN, CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. Audio Preprocessing (Identical to TESSDataset pipeline)
    sr = 16000
    max_duration = 3.0
    max_len = int(sr * max_duration)  # 48000 samples

    # Load audio and trim silence (top_db=20 matches your dataset setup)
    audio, _ = librosa.load(audio_path, sr=sr)
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Standardize length (padding / truncation)
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = np.pad(audio, (0, max_len - len(audio)))

    # 3. Feature Extraction (MFCCs + Deltas)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    d1 = librosa.feature.delta(mfcc, axis=1)
    d2 = librosa.feature.delta(mfcc, order=2, axis=1)

    # Combine and Transpose to fit (Sequence_Length, Feature_Dimension)
    feat = np.concatenate([mfcc, d1, d2], axis=0).T
    
    # Z-score Normalization (Crucial: Must match training scale)
    feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)

    # Convert to Tensor and add Batch Dimension: Shape becomes (1, Sequence_Length, Features)
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)

    # 4. Model Inference
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]

    predicted_idx = np.argmax(prob)
    
    return {
        "emotion": EMOTIONS[predicted_idx],
        "confidence": float(prob[predicted_idx]),
        "all_probabilities": {EMOTIONS[i]: float(prob[i]) for i in range(len(EMOTIONS))}
    }


if __name__ == "__main__":
    # Example usage: You can test it on a sample from your drive or a custom clip
    # Make sure to give a path to a valid .wav or .ogg file
    sample_file = "/content/drive/MyDrive/Multimodal Emotion Recognition/data/my_audio.ogg"
    
    if os.path.exists(sample_file):
        print(f"Running inference on: {os.path.basename(sample_file)}")
        result = predict(sample_file)
        
        print("\n--- Inference Results ---")
        print(f"Predicted Emotion : {result['emotion'].upper()}")
        print(f"Confidence Score  : {result['confidence'] * 100:.2f}%")
        print("\nFull breakdown:")
        for emo, p in result['all_probabilities'].items():
            print(f"  {emo:10s}: {p * 100:.2f}%")
    else:
        print(f"To test inference, update the 'sample_file' path with a real file. File not found at: {sample_file}")
