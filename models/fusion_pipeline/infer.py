import os
import sys
import glob
import torch
import librosa
import numpy as np
from transformers import BertTokenizer

# Routing paths dynamically
search_speech = glob.glob('/content/drive/**/speech_pipeline/model_utils.py', recursive=True)
search_text = glob.glob('/content/drive/**/text_pipeline/text_model_utils.py', recursive=True)
SPEECH_DIR = os.path.dirname(search_speech[0])
TEXT_DIR = os.path.dirname(search_text[0])

if SPEECH_DIR not in sys.path: sys.path.append(SPEECH_DIR)
if TEXT_DIR not in sys.path: sys.path.append(TEXT_DIR)

from model_utils import SpeechEmotionModel
from text_emotion_model import TextEmotionModel # adjustment match matching layout name
from text_model_utils import TextEmotionModel
from fusion_model_utils import MultimodalFusionModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Pleasant Surprise', 'Sad']

# Instantiate raw baselines
speech_base = SpeechEmotionModel(input_size=120, hidden=128, num_classes=7).to(device)
text_base = TextEmotionModel(num_classes=7).to(device)
model = MultimodalFusionModel(speech_base, text_base, num_classes=7).to(device)

# Load Fusion Weights
fusion_weights = glob.glob('/content/drive/**/fusion_pipeline/best_fusion_model.pth', recursive=True)[0]
model.load_state_dict(torch.load(fusion_weights, map_location=device))
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def run_live_inference(audio_file_path, text_transcript):
    print("\n🔄 Running Cross-Modality Late Fusion Feature Parsing...")
    
    # 1. Preprocess Spoken Target Signal
    y, sr = librosa.load(audio_file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    audio_features = np.concatenate((mfcc, delta, delta2), axis=0).T
    audio_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 2. Preprocess Literal Transcript String
    inputs = tokenizer(text_transcript, padding='max_length', truncation=True, max_length=16, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # 3. Process Joint Decision Pass
    with torch.no_grad():
        logits = model(audio_tensor, input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        max_idx = torch.argmax(probabilities).item()
        
    print("="*45)
    print(f"🎙️ Audio File Evaluated : {os.path.basename(audio_file_path)}")
    print(f"🔤 Transcribed Text Used : \"{text_transcript}\"")
    print("-"*45)
    print(f"🏆 Final Multimodal Label : {EMOTIONS[max_idx]} ({probabilities[max_idx]*100:.2f}% Confidence)")
    print("="*45)

# Example execution template:
# Pass a real raw file and type its semantic match to test boundary corrections
# run_live_inference("/content/drive/MyDrive/.../YAF_kick_angry.wav", "The word is kick")
