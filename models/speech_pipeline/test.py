import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model_utils import TESSDataset, SpeechEmotionModel

MODEL_DIR = "../models"
# Resolving directory structure layout to drop files cleanly in your main project folder
RESULT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "models", "Results"))

N_MFCC = 40
INPUT = N_MFCC * 3
HIDDEN = 128
CLASSES = 7
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

def extract_layers(model, loader, device):
    """Extracts targets along with Temporal and Contextual representations."""
    X_ctx, X_temp, y_true, y_pred = [], [], [], []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, lstm_out, ctx, _ = model(x, return_features=True)
            preds = torch.argmax(logits, dim=1)
            
            # Temporal Compression: Mean-pool representations across sequence time-steps [cite: 74]
            temp_pooled = torch.mean(lstm_out, dim=1)

            X_ctx.append(ctx.cpu().numpy())
            X_temp.append(temp_pooled.cpu().numpy())
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())

    return np.concatenate(X_ctx), np.concatenate(X_temp), np.array(y_true), np.array(y_pred)

def generate_tsne_plot(features, labels, title, filename):
    print(f"Computing t-SNE projections for: {title}...")
    tsne = TSNE(n_components=2, perplexity=25, random_state=42, n_iter=1000)
    projections = tsne.fit_transform(features)

    plt.figure(figsize=(9, 7))
    for i, emo in enumerate(EMOTIONS):
        mask = labels == i
        plt.scatter(projections[mask, 0], projections[mask, 1], label=emo, s=15, alpha=0.8)
        
    plt.title(f"TESS - {title} Cluster Space")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(RESULT_DIR, filename)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"✅ Saved plot to: {save_path}")

def test():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = TESSDataset()
    
    # Load frozen test tokens saved during training execution
    indices_path = os.path.join(MODEL_DIR, "test_indices.pt")
    weights_path = os.path.join(MODEL_DIR, "speech_model_best.pth")
    
    if not (os.path.exists(indices_path) and os.path.exists(weights_path)):
        raise FileNotFoundError("❌ Disconnected pipelines! Run train.py completely before execution.")
        
    te_idx = torch.load(indices_path)
    loader = DataLoader(Subset(dataset, te_idx), batch_size=32, shuffle=False)

    model = SpeechEmotionModel(INPUT, HIDDEN, CLASSES).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    X_ctx, X_temp, y_true, y_pred = extract_layers(model, loader, device)

    # Text Analysis Logs [cite: 61, 68]
    print("\n" + "="*60)
    print("           SPEECH EVALUATION CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))
    print("="*60)

    # Graph 1 & 2 Generation: t-SNE Maps [cite: 73, 74, 75]
    generate_tsne_plot(X_temp, y_true, "Temporal Modelling Block (LSTM)", "tsne_temporal.png")
    generate_tsne_plot(X_ctx, y_true, "Contextual Modelling Block (Attention)", "tsne_contextual.png")

    # Graph 3 Generation: Confusion Matrix [cite: 71]
    print("Plotting Performance Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title("Speech Model Matrix Confusions")
    plt.ylabel("Actual Labels")
    plt.xlabel("Predicted Labels")
    plt.tight_layout()
    
    matrix_path = os.path.join(RESULT_DIR, "cm_speech.png")
    plt.savefig(matrix_path, dpi=200)
    plt.close()
    print(f"✅ Saved matrix map to: {matrix_path}\n")

if __name__ == "__main__":
    test()
