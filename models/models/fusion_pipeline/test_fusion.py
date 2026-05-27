import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- CORE PATH INJECTION ---
# Inject the root 'models' directory to the system path so packages import cleanly
MODELS_PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if MODELS_PARENT_DIR not in sys.path:
    sys.path.insert(0, MODELS_PARENT_DIR)

# Import components using absolute folder namespaces to avoid local script shadowing
import speech_pipeline.model_utils as speech_utils
import text_pipeline.model_utils as text_utils
from fusion_model_utils import MultimodalFusionModel

# --- EXACT DIRECTORY STRUCTURE CONFIGURATIONS ---
FUSION_WEIGHTS = "./best_fusion_model.pth"
INDICES_PATH = os.path.abspath(os.path.join(MODELS_PARENT_DIR, "speech_pipeline", "test_indices.pt"))
RESULT_DIR = os.path.abspath(os.path.join(MODELS_PARENT_DIR, "..", "results"))

AUDIO_DATA_DIR = "/content/drive/MyDrive/Multimodal Emotion Recognition/data/TESS"
TEXT_DATA_DIR = "/content/local_tess"

NUM_CLASSES = 7
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
CLASS_IDS = [0, 1, 2, 3, 4, 5, 6]

class MultimodalEvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, audio_ds, text_ds):
        self.audio_ds = audio_ds
        self.text_ds = text_ds

    def __len__(self):
        return len(self.audio_ds)

    def __getitem__(self, idx):
        speech_feat, label = self.audio_ds[idx]
        input_ids, attention_mask, text_label = self.text_ds[idx]
        return speech_feat, input_ids, attention_mask, label

def extract_fusion_layers(model, loader, device):
    X_fused, y_true, y_pred = [], [], []
    
    with torch.no_grad():
        for speech_features, input_ids, attention_mask, labels in loader:
            speech_features = speech_features.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            _, _, speech_context, _ = model.speech_backbone(speech_features, return_features=True)
            bert_outputs = model.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
            text_context = bert_outputs.pooler_output
            
            s_proj = torch.relu(model.speech_proj(speech_context))
            t_proj = torch.relu(model.text_proj(text_context))
            
            combined_proj = torch.cat((s_proj, t_proj), dim=1)
            gate_weight = model.gate(combined_proj)
            
            fused_vector = gate_weight * s_proj + (1.0 - gate_weight) * t_proj
            
            logits = model.classifier(fused_vector)
            predictions = torch.argmax(logits, dim=1)
            
            X_fused.append(fused_vector.cpu().numpy())
            y_true.extend(labels.numpy())
            y_pred.extend(predictions.cpu().numpy())
            
    return np.concatenate(X_fused), np.array(y_true), np.array(y_pred)

def generate_fusion_tsne(features, labels):
    print("Computing t-SNE projections for the Multimodal Fusion block...")
    tsne = TSNE(n_components=2, perplexity=25, random_state=42, n_iter=1000)
    projections = tsne.fit_transform(features)

    plt.figure(figsize=(9, 7))
    for i, emo in enumerate(EMOTIONS):
        mask = labels == i
        plt.scatter(projections[mask, 0], projections[mask, 1], label=emo, s=15, alpha=0.8)
        
    plt.title("TESS - Multimodal Fusion Cluster Space")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(RESULT_DIR, "tsne_fusion.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"✅ Saved Fusion cluster graph to: {save_path}")

def evaluate_fusion():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Multimodal Fusion Evaluation on: {device}")
    
    # Initialize sub-models with namespace-mapped weights
    speech_base = speech_utils.SpeechEmotionModel(input_size=120, hidden=128, num_classes=7)
    text_base = text_utils.TextEmotionModel(num_classes=7)
    
    model = MultimodalFusionModel(speech_base, text_base, num_classes=7).to(device)
    if not os.path.exists(FUSION_WEIGHTS):
        raise FileNotFoundError(f"❌ Missing fusion weights checkpoint at: {FUSION_WEIGHTS}")
    model.load_state_dict(torch.load(FUSION_WEIGHTS, map_location=device))
    model.eval()
    
    audio_dataset = speech_utils.TESSDataset(data_dir=AUDIO_DATA_DIR)
    text_dataset = text_utils.TESSTextDataset(data_dir=TEXT_DATA_DIR)
    combined_dataset = MultimodalEvaluationDataset(audio_dataset, text_dataset)
    
    if os.path.exists(INDICES_PATH):
        print(f"📦 Loading frozen test split tokens from: {INDICES_PATH}")
        te_idx = torch.load(INDICES_PATH)
        eval_subset = Subset(combined_dataset, te_idx)
    else:
        print("⚠️ Warning: Shared test indices file not found. Falling back to whole dataset...")
        eval_subset = combined_dataset

    loader = DataLoader(eval_subset, batch_size=32, shuffle=False)
    
    X_fused, y_true, y_pred = extract_fusion_layers(model, loader, device)
    
    report_text = classification_report(y_true, y_pred, labels=CLASS_IDS, target_names=EMOTIONS, zero_division=0)
    print("\n" + "="*60)
    print("         MULTIMODAL FUSION EVALUATION REPORT")
    print("="*60)
    print(report_text)
    print("="*60)
    
    report_path = os.path.join(RESULT_DIR, "fusion_report.txt")
    with open(report_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("         MULTIMODAL FUSION EVALUATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(report_text)
        f.write("="*60 + "\n")
    print(f"✅ Saved fusion variant accuracy table to: {report_path}")
    
    generate_fusion_tsne(X_fused, y_true)
    
    print("Plotting Fusion Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title("Fusion Model Matrix Confusions")
    plt.ylabel("Actual Labels")
    plt.xlabel("Predicted Labels")
    plt.tight_layout()
    
    matrix_path = os.path.join(RESULT_DIR, "cm_fusion.png")
    plt.savefig(matrix_path, dpi=200)
    plt.close()
    print(f"✅ Saved fusion matrix map to: {matrix_path}\n")

if __name__ == "__main__":
    evaluate_fusion()
