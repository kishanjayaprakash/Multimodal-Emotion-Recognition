import sys
import os
import glob
import librosa
import numpy as np
import torch
import torch.nn.functional as F

# =====================================================================
# 🚨 ONLY CHANGE THIS FILE NAME TO TEST DIFFERENT AUDIOS!
# =====================================================================
TARGET_AUDIO_FILE = "03-01-04-02-01-01-12.wav"  # Your sad file name

# =====================================================================
# 1. DYNAMIC FILE DISCOVERY
# =====================================================================
print("🔄 Autodetecting project paths...")

search_weights = glob.glob('/content/drive/**/speech_model_best.pth', recursive=True)
search_audio = glob.glob(f'/content/drive/**/{TARGET_AUDIO_FILE}', recursive=True)

if not search_weights:
    raise FileNotFoundError("❌ Could not find 'speech_model_best.pth' in your Drive.")
if not search_audio:
    raise FileNotFoundError(f"❌ Could not find the file '{TARGET_AUDIO_FILE}' in your Drive. Check the filename spelling!")

MODEL_WEIGHTS_PATH = search_weights[0]
FILE_PATH = search_audio[0]
PROJECT_PATH = os.path.dirname(FILE_PATH)

if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from model_utils import SpeechEmotionModel

# =====================================================================
# 2. MODEL CONFIGURATION (Pulled straight from your model_utils.py)
# =====================================================================
INPUT_DIM = 40 * 3   # 120 features
HIDDEN_DIM = 128
CLASSES_NUM = 7
