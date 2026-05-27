import os
import sys
import random
import glob
import librosa
import numpy as np
import torch
import torch.nn as nn

# ----------------- ENVIRONMENT PATH CONFIG -----------------
SPEECH_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_MODELS_DIR = os.path.abspath(os.path.join(SPEECH_DIR, ".."))

if SPEECH_DIR not in sys.path:
    sys.path.insert(0, SPEECH_DIR)
if PARENT_MODELS_DIR not in sys.path:
    sys.path.insert(0, PARENT_MODELS_DIR)

from model_utils import SpeechEmotionModel

# 💡 SPECIFIC TARGETING: Look strictly for files containing 'speech' to prevent loading fusion weights
DATA_DIR = os.path.abspath(os.path.join(PARENT_MODELS_DIR, "data\\TESS"))
MODEL_PATH = os.path.abspath(os.path.join(PARENT_MODELS_DIR, "speech_model_best.pth"))

if not os.path.exists(MODEL_PATH):
    # Fallback to look for alternative variations like 'best_speech_model (4).pth'
    speech_weights_variants = glob.glob(os.path.join(PARENT_MODELS_DIR, "*speech*.pth"))
    if speech_weights_variants:
        MODEL_PATH = speech_weights_variants[0]
    else:
        # Final safety check fallback if no name hints match
        MODEL_PATH = os.path.abspath(os.path.join(PARENT_MODELS_DIR, "best_speech_model (4).pth"))

N_MFCC = 40
INPUT = 120  # Explicitly matches your true trained Bi-LSTM configuration parameters
HIDDEN = 128
CLASSES = 7

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']


def predict(audio_path):
    """Loads a single audio file, extracts features, and predicts the emotion."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Trained speech model weights file missing at: {MODEL_PATH}")

    print(f"🤖 Initializing Model Architecture and loading weights from: {os.path.basename(MODEL_PATH)}")
    model = SpeechEmotionModel(INPUT, HIDDEN, CLASSES).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # 💡 REMAP STRUCTURAL LAYER NAMES IF AN OLD CHECKPOINT IS LOADED
    adapted_checkpoint = {}
    for k, v in checkpoint.items():
        if k == "fc.weight":
            adapted_checkpoint["classifier.3.weight"] = v
        elif k == "fc.bias":
            adapted_checkpoint["classifier.3.bias"] = v
        else:
            adapted_checkpoint[k] = v

    model.load_state_dict(adapted_checkpoint, strict=False)
    model.eval()

    # Audio Preprocessing
    sr = 16000
    max_duration = 3.0
    max_len = int(sr * max_duration)

    audio, _ = librosa.load(audio_path, sr=sr)
    audio, _ = librosa.effects.trim(audio, top_db=20)

    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = np.pad(audio, (0, max_len - len(audio)))

    # Feature Extraction (MFCCs + Deltas)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    d1 = librosa.feature.delta(mfcc, axis=1)
    d2 = librosa.feature.delta(mfcc, order=2, axis=1)

    feat = np.concatenate([mfcc, d1, d2], axis=0).T
    
    # Z-score Normalization
    feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)

    # Model Inference
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
    print("🔄 Initializing Single-File Inference Script...")
    
    local_wavs = glob.glob(os.path.join(DATA_DIR, "**\\*.wav"), recursive=True)
    
    if local_wavs:
        sample_file = random.choice(local_wavs)
        print("✅ Automatically pulled random sample from dataset.")
    else:
        sample_file = "./my_audio.wav"

    if os.path.exists(sample_file):
        print(f"🎬 Running inference on audio track: {os.path.basename(sample_file)}")
        print(f"📁 Absolute Source File Path: {sample_file}")
        
        result = predict(sample_file)
        
        print("\n" + "="*45)
        print("          LIVE SPEECH INFERENCE CONSOLE")
        print("="*45)
        print(f"🔮 Predicted Emotion : {result['emotion'].upper()}")
        print(f"🎯 Confidence Score  : {result['confidence'] * 100:.2f}%")
        print("-"*45)
        print("📊 Distribution Matrix Breakdown:")
        for emo, p in result['all_probabilities'].items():
            print(f"  • {emo:18s}: {p * 100:.2f}%")
        print("="*45 + "\n")
    else:
        print(f"❌ Error: Could not find dataset paths at: {DATA_DIR}")