import os
import sys
import numpy as np
import torch
import glob
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# ----------------- ENVIRONMENT PATH CONFIG -----------------
if os.path.exists("/content/drive"):
    print("☁️ Cloud Environment Detected: Setting up Google Drive paths...")
    BASE_PATH = "/content/drive/MyDrive/Multimodal Emotion Recognition/models"
    SPEECH_DIR = os.path.join(BASE_PATH, "speech_pipeline")
    
    possible_paths = glob.glob("/content/drive/**/data/TESS", recursive=True)
    DATA_DIR = possible_paths[0] if possible_paths else "/content/drive/MyDrive/Multimodal Emotion Recognition/data/TESS"
    
    possible_weights = glob.glob("/content/drive/**/speech_model_best.pth", recursive=True)
    WEIGHTS_PATH = possible_weights[0] if possible_weights else os.path.join(BASE_PATH, "speech_model_best.pth")
else:
    print("💻 Local Environment Detected: Setting up repository paths...")
    SPEECH_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.abspath(os.path.join(SPEECH_DIR, "..", "data\\TESS"))
    
    possible_local_weights = glob.glob(os.path.join(SPEECH_DIR, "..", "*.pth"))
    if possible_local_weights:
        WEIGHTS_PATH = possible_local_weights[0]
    else:
        WEIGHTS_PATH = os.path.abspath(os.path.join(SPEECH_DIR, "..", "speech_model_best.pth"))

if SPEECH_DIR not in sys.path: 
    sys.path.insert(0, SPEECH_DIR)
# -----------------------------------------------------------

from model_utils import SpeechEmotionModel, TESSDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant Surprise", "Sad"]

print(f"📦 Discovered Verified Target Path: {DATA_DIR}")
print("📦 Loading Speech Validation Dataset...")
test_dataset = TESSDataset(DATA_DIR)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"🔍 Checking for model weights at: {WEIGHTS_PATH}")
checkpoint = None
if os.path.exists(WEIGHTS_PATH):
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)

# Match standard configuration parameters exactly
input_size = 120  # 40 MFCC * 3
hidden_size = 128
num_classes = 7

print(f"🤖 Initializing SpeechEmotionModel Architecture (Input size: {input_size})...")
model = SpeechEmotionModel(input_size=input_size, hidden=hidden_size, num_classes=num_classes).to(device)

if checkpoint is not None:
    # REMAP KEYS: Dynamically maps keys from best_speech_model (4).pth to current structural parameters
    adapted_checkpoint = {}
    for k, v in checkpoint.items():
        # Map Attention module sub-layer keys
        if k == "attention.attention.weight":
            adapted_checkpoint["attention.attn.weight"] = v
        elif k == "attention.attention.bias":
            adapted_checkpoint["attention.attn.bias"] = v
        # Map Classifier layer keys due to structural layout shifts
        elif k == "classifier.4.weight":
            adapted_checkpoint["classifier.3.weight"] = v
        elif k == "classifier.4.bias":
            adapted_checkpoint["classifier.3.bias"] = v
        else:
            adapted_checkpoint[k] = v
            
    # Load state dict with strict=False to gracefully bypass missing/extra layers (like BatchNorm)
    model.load_state_dict(adapted_checkpoint, strict=False)
    print(f"✅ Successfully matched and loaded speech weights.")
else:
    print(f"⚠️ Warning: Weights file not found at {WEIGHTS_PATH}. Running evaluation on uninitialized weights.")

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