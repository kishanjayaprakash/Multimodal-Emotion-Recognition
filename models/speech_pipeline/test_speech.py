import os
import sys
import numpy as np
import torch
import glob
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

BASE_PATH = "/content/drive/MyDrive/Multimodal Emotion Recognition/models"
SPEECH_DIR = os.path.join(BASE_PATH, "speech_pipeline")
if SPEECH_DIR not in sys.path: sys.path.insert(0, SPEECH_DIR)

from model_utils import SpeechEmotionModel, TESSDataset

# Robust path discovery to protect against cloud string truncation
possible_paths = glob.glob("/content/drive/**/data/TESS", recursive=True)
DATA_DIR = possible_paths[0] if possible_paths else "/content/drive/MyDrive/Multimodal Emotion Recognition/data/TESS"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant Surprise", "Sad"]

print(f"📦 Discovered Verified Target Path: {DATA_DIR}")
print("📦 Loading Speech Validation Dataset...")
test_dataset = TESSDataset(DATA_DIR)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("🤖 Initializing Bi-LSTM + Attention Architecture...")
model = SpeechEmotionModel(input_size=120, hidden=128, num_classes=7).to(device)

SPEECH_WEIGHTS = glob.glob("/content/drive/**/speech_model_best.pth", recursive=True)
if SPEECH_WEIGHTS:
    model.load_state_dict(torch.load(SPEECH_WEIGHTS[0], map_location=device))
    print(f"✅ Pre-loaded speech weights from: {SPEECH_WEIGHTS[0]}")

model.eval()
y_true, y_pred = [], []

print("🧪 Commencing Speech Evaluation Loop...")
with torch.no_grad():
    for audio_in, labels in test_loader:
        audio_in = audio_in.to(device)
        outputs = model(audio_in)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

y_true, y_pred = np.array(y_true), np.array(y_pred)
report_dict = classification_report(y_true, y_pred, target_names=EMOTIONS, output_dict=True, zero_division=0)

report_file = os.path.join(SPEECH_DIR, "speech_per_class_accuracy.txt")
with open(report_file, "w") as out:
    out.write("=============================================\n")
    out.write("     SPEECH PIPELINE PER-CLASS ACCURACY\n")
    out.write("=============================================\n")
    for emotion in EMOTIONS:
        class_acc = report_dict[emotion]["recall"] * 100
        out.write(f"• {emotion:<20} Accuracy : {class_acc:.2f}%\n")
    out.write("=============================================\n")

print(f"🎉 Speech per-class accuracy report written to:\n📁 {report_file}")
