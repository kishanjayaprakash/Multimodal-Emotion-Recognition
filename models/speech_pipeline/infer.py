import sys
import os
import glob
import librosa
import numpy as np
import torch
import torch.nn.functional as F

print("🔄 Autodetecting project paths...")

# 1. Automated file discovery
search_weights = glob.glob('/content/drive/**/speech_model_best.pth', recursive=True)
search_audio = glob.glob('/content/drive/**/03-01-04-02-01-01-12.wav', recursive=True)

if not search_weights or not search_audio:
    raise FileNotFoundError("Could not find weights or audio files.")

MODEL_WEIGHTS_PATH = search_weights[0]
FILE_PATH = search_audio[0]
PROJECT_PATH = os.path.dirname(FILE_PATH)

if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from model_utils import SpeechEmotionModel

# =====================================================================
# 2. EXACT MODEL CONFIGURATION
# =====================================================================
INPUT_DIM = 120   # 40 MFCCs * 3
HIDDEN_DIM = 128
CLASSES_NUM = 7
