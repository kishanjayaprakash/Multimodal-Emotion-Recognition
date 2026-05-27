import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer
from model_utils import TESSTextDataset, TextEmotionModel

# --- EXACT PATHS BASED ON YOUR DIRECTORY IMAGE ---
MODEL_PATH = "./best_text_model.pth"
INDICES_PATH = "./test_indices.pt"
RESULT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "results"))
DATA_DIR = "/content/local_tess" 

NUM_CLASSES = 7
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
# Explicit class integer mapping matching scikit-learn positions
CLASS_IDS = [0, 1, 2, 3, 4, 5, 6]

def evaluate_dataset():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Text Evaluation on device: {device}")
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError("❌ Fast local data folder missing. Make sure you ran the copy block!")
        
    dataset = TESSTextDataset(DATA_DIR)
    
    if os.path.exists(INDICES_PATH):
        print(f"📦 Loading frozen test split tokens from: {INDICES_PATH}")
        te_idx = torch.load(INDICES_PATH)
        eval_subset = Subset(dataset, te_idx)
    else:
        print("⚠️ Warning: test_indices.pt not found. Evaluating on full dataset...")
        eval_subset = dataset

    loader = DataLoader(eval_subset, batch_size=32, shuffle=False)

    model = TextEmotionModel(NUM_CLASSES).to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Missing text model weights file at: {MODEL_PATH}")
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    print("🏃 Running validation inputs through BERT layers...")
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(predictions.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # CRITICAL FIX: Added labels=CLASS_IDS to prevent class count mismatches
    report_text = classification_report(y_true, y_pred, labels=CLASS_IDS, target_names=EMOTIONS, zero_division=0)
    
    print("\n" + "="*60)
    print("           TEXT EVALUATION CLASSIFICATION REPORT")
    print("="*60)
    print(report_text)
    print("="*60)

    # Save variant accuracy table text file directly to results folder
    report_path = os.path.join(RESULT_DIR, "text_report.txt")
    with open(report_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("           TEXT EVALUATION CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(report_text)
        f.write("="*60 + "\n")
    print(f"✅ Saved text variant accuracy table to: {report_path}")

    print("Plotting Text Performance Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title("Text Model Matrix Confusions")
    plt.ylabel("Actual Labels")
    plt.xlabel("Predicted Labels")
    plt.tight_layout()
    
    matrix_path = os.path.join(RESULT_DIR, "cm_text.png")
    plt.savefig(matrix_path, dpi=200)
    plt.close()
    print(f"✅ Saved text matrix map to: {matrix_path}\n")

if __name__ == "__main__":
    evaluate_dataset()
