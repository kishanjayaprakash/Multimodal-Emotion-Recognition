import os
import sys
import torch
from transformers import BertTokenizer

# ----------------- ENVIRONMENT PATH CONFIG -----------------
# Establishes context bounds based on file position rather than working directory
FUSION_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_MODELS_DIR = os.path.abspath(os.path.join(FUSION_DIR, ".."))

TEXT_PIPELINE_DIR = os.path.join(PARENT_MODELS_DIR, "text_pipeline")
SPEECH_PIPELINE_DIR = os.path.join(PARENT_MODELS_DIR, "speech_pipeline")

if FUSION_DIR not in sys.path: sys.path.insert(0, FUSION_DIR)
if PARENT_MODELS_DIR not in sys.path: sys.path.insert(0, PARENT_MODELS_DIR)
if TEXT_PIPELINE_DIR not in sys.path: sys.path.insert(0, TEXT_PIPELINE_DIR)
if SPEECH_PIPELINE_DIR not in sys.path: sys.path.insert(0, SPEECH_PIPELINE_DIR)

# Core imports aligned to your actual file architecture naming layouts
import model_utils as speech_utils
import text_model_utils as text_utils
# -----------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant surprise", "sad"]

print("🔄 Initializing Full Dynamic Multimodal Inference Rig...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Set configurations matching your true architecture configurations
speech_base = speech_utils.SpeechEmotionModel(input_size=120, hidden=128, num_classes=7)
text_base = text_utils.TextEmotionModel(num_classes=7)

# Dynamically resolve import maps for MultimodalFusionModel if it sits inside a utilities file
try:
    from fusion_model_utils import MultimodalFusionModel
except ImportError:
    from model_utils import MultimodalFusionModel

model = MultimodalFusionModel(speech_base, text_base, num_classes=7).to(device)

# Local checkout strategy for weight files paths
WEIGHTS_PATH = "./best_fusion_model.pth"
if not os.path.exists(WEIGHTS_PATH):
    local_weights_fallback = os.path.abspath(os.path.join(PARENT_MODELS_DIR, "best_fusion_model.pth"))
    if os.path.exists(local_weights_fallback):
        WEIGHTS_PATH = local_weights_fallback
    else:
        raise FileNotFoundError(f"❌ Checkpoint missing at: {WEIGHTS_PATH}")

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
print(f"✅ Combined fusion weights loaded successfully.")
model.eval()

# Local data verification check matching your dataset sidebars layout
AUDIO_DATA_DIR = "/content/drive/MyDrive/Multimodal Emotion Recognition/data/TESS"
if not os.path.exists(AUDIO_DATA_DIR):
    local_data_fallback = os.path.abspath(os.path.join(PARENT_MODELS_DIR, "data\\TESS"))
    if os.path.exists(local_data_fallback):
        AUDIO_DATA_DIR = local_data_fallback
    else:
        raise FileNotFoundError(f"❌ Verification dataset paths missing. Checked: {AUDIO_DATA_DIR}")

audio_dataset = speech_utils.TESSDataset(data_dir=AUDIO_DATA_DIR)

print("\n" + "="*60)
print("             DYNAMIC AUDIO + TEXT FUSION CONSOLE")
print("="*60)
print("Type 'exit' to terminate interface loop.\n")

while True:
    index_input = input(f"Enter an audio index to load (0 to {len(audio_dataset)-1}) ➔ ")
    if index_input.lower() == 'exit':
        break
    try:
        idx = int(index_input)
        speech_features, true_label_idx = audio_dataset[idx]
        speech_tensor = speech_features.unsqueeze(0).to(device)
        print(f"🔊 Loaded audio track index {idx}. True Audio Class: {EMOTIONS[true_label_idx].upper()}")
    except Exception as e:
        print(f"❌ Invalid index footprint. Processing check failed: {e}")
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
        
        # Pull hidden features context pooling states gracefully matching BERT representation blocks
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