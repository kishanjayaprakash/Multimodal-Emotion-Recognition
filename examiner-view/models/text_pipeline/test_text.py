import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer

# --- EXACT PATHS BASED ON YOUR DIRECTORY IMAGE ---
MODEL_PATH = "./best_text_model.pth"
INDICES_PATH = "./test_indices.pt"
RESULT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "results"))
DATA_DIR = "/content/local_tess" 

NUM_CLASSES = 7
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
CLASS_IDS = [0, 1, 2, 3, 4, 5, 6]

def evaluate_dataset():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Text Evaluation on device: {device}")
    
    # Locate the folder where this script is running (models/text_pipeline)
    TEXT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_MODELS_DIR = os.path.abspath(os.path.join(TEXT_DIR, ".."))
    
    # Inject both folders into the python path search system
    if TEXT_DIR not in sys.path:
        sys.path.insert(0, TEXT_DIR)
    if PARENT_MODELS_DIR not in sys.path:
        sys.path.insert(0, PARENT_MODELS_DIR)
        
    # 💡 CHANGED IMPORT TO MATCH YOUR FILE: text_model_utils
    from text_model_utils import TESSTextDataset, TextEmotionModel
    
    if not os.path.exists(DATA_DIR):
        fallback_data = os.path.abspath(os.path.join(PARENT_MODELS_DIR, "data\\TESS"))
        if os.path.exists(fallback_data):
            current_data_dir = fallback_data
        else:
            raise FileNotFoundError(f"❌ Data folder missing. Verified paths checked: {DATA_DIR} or {fallback_data}")
    else:
        current_data_dir = DATA_DIR
        
    dataset = TESSTextDataset(current_data_dir)
    
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
        local_weights = os.path.abspath(os.path.join(PARENT_MODELS_DIR, "best_text_model.pth"))
        if os.path.exists(local_weights):
            current_model_path = local_weights
        else:
            raise FileNotFoundError(f"❌ Missing text model weights file at: {MODEL_PATH}")
    else:
        current_model_path = MODEL_PATH
        
    model.load_state_dict(torch.load(current_model_path, map_location=device))
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

    report_text = classification_report(y_true, y_pred, labels=CLASS_IDS, target_names=EMOTIONS, zero_division=0)
    
    print("\n" + "="*60)
    print("           TEXT EVALUATION CLASSIFICATION REPORT")
    print("="*60)
    print(report_text)
    print("="*60)

    report_path = os.path.join(RESULT_DIR, "text_report.txt")
    with open(report_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("           TEXT EVALUATION CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(report_text)
        f.write("="*60 + "\n")
    print(f"✅ Saved text variant accuracy table to: {report_path}")

    print("Plotting Text Performance Confusion Matrix...")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
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
    except Exception as e:
        print(f"\n⚠️ Matplotlib initialization warning bypassed safely.")
        print("✅ Numerical metrics computed and saved perfectly.")

if __name__ == "__main__":
    evaluate_dataset()