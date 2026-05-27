import os
import sys
import torch
from transformers import BertTokenizer

# Secure parent namespaces
MODELS_PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if MODELS_PARENT_DIR not in sys.path:
    sys.path.insert(0, MODELS_PARENT_DIR)

import speech_pipeline.model_utils as speech_utils
import text_pipeline.model_utils as text_utils
from fusion_model_utils import MultimodalFusionModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant surprise", "sad"]

print("🔄 Initializing Full Dynamic Multimodal Inference Rig...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
speech_base = speech_utils.SpeechEmotionModel(input_size=120, hidden=128, num_classes=7)
text_base = text_utils.TextEmotionModel(num_classes=7)
model = MultimodalFusionModel(speech_base, text_base, num_classes=7).to(device)

WEIGHTS_PATH = "./best_fusion_model.pth"
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print(f"✅ Combined fusion weights loaded.")
else:
    raise FileNotFoundError(f"❌ Checkpoint missing at: {WEIGHTS_PATH}")

model.eval()

# Load the dataset arrays
AUDIO_DATA_DIR = "/content/drive/MyDrive/Multimodal Emotion Recognition/data/TESS"
audio_dataset = speech_utils.TESSDataset(data_dir=AUDIO_DATA_DIR)

print("\n" + "="*60)
print("             DYNAMIC AUDIO + TEXT FUSION CONSOLE")
print("="*60)
print("Type 'exit' to quit.\n")

while True:
    index_input = input(f"Enter an audio index to load (0 to {len(audio_dataset)-1}) ➔ ")
    if index_input.lower() == 'exit':
        break
    try:
        idx = int(index_input)
        speech_features, true_label_idx = audio_dataset[idx]
        speech_tensor = speech_features.unsqueeze(0).to(device)
        print(f"🔊 Loaded audio track index {idx}. True Audio Class: {EMOTIONS[true_label_idx].upper()}")
    except Exception:
        print("❌ Invalid index. Please enter a valid integer.")
        continue

    user_text = input("Enter companion text string ➔ ")
    if user_text.lower() == 'exit':
        break

    encoding = tokenizer(
        f"The word is {user_text.lower().strip()}",
        padding='max_length',
        truncation=True,
        max_length=16,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        _, _, speech_context, _ = model.speech_backbone(speech_tensor, return_features=True)
        bert_outputs = model.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_context = bert_outputs.pooler_output
        
        s_proj = torch.relu(model.speech_proj(speech_context))
        t_proj = torch.relu(model.text_proj(text_context))
        
        combined_proj = torch.cat((s_proj, t_proj), dim=1)
        gate_weight = model.gate(combined_proj).item() 
        
        fused_vector = gate_weight * s_proj + (1.0 - gate_weight) * t_proj
        logits = model.classifier(fused_vector)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        prediction_idx = torch.argmax(probabilities).item()

    print(f"\n📊 Gate Breakdown:")
    print(f"   ↳ Speech Importance Weight (g) : {gate_weight * 100:.1f}%")
    print(f"   ↳ Text Importance Weight (1-g) : {(1.0 - gate_weight) * 100:.1f}%")
    print(f"🔮 Final Multimodal Prediction    : {EMOTIONS[prediction_idx].upper()} ({probabilities[prediction_idx]*100:.2f}% Confidence)")
    print("-" * 60)
