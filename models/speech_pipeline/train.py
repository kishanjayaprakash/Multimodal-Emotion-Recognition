import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from model_utils import TESSDataset, SpeechEmotionModel

# ---------------- CONFIG ----------------
BATCH = 32
INPUT = 40 * 3  # N_MFCC (40) * 3 (MFCC + Delta + Delta2)
HIDDEN = 128
CLASSES = 7
LR = 0.001
EPOCHS = 20

def get_lexical_split(dataset, test_size=0.2):
    """
    Parses TESS filenames, extracts the unique spoken words, and splits
    the data cleanly by word to prevent acoustic data leakage.
    """
    words = []
    
    for path in dataset.file_paths:
        filename = os.path.basename(path).lower().replace(".wav", "")
        parts = filename.split("_")
        
        # Standard TESS format: YAF_back_angry.wav -> word is index 1
        # If your filenames are grouped as 'OAF_angry_back.wav', change parts[1] to parts[2]
        if len(parts) >= 3:
            word = parts[1].strip()
        else:
            word = parts[0].strip()
            
        words.append(word)
        
    words = np.array(words)
    unique_words = np.unique(words)
    
    print("\n" + "="*50)
    print("                SPLIT ENGINE DEBUG")
    print("="*50)
    print(f"[Split Engine] Total unique words found: {len(unique_words)}")
    print(f"[Split Engine] Sample words extracted: {list(unique_words[:10])}")
    print("-"*50)
    
    # CRITICAL CHECK: If you see 'happy', 'angry', or 'sad' in the sample words above,
    # it means your filename indexing is wrong and causing the leakage!

    # Split the unique dictionary words (80/20)
    tr_words, te_words = train_test_split(
        unique_words, 
        test_size=test_size, 
        random_state=42
    )
    
    # Map word allocations back to file indices
    tr_idx = [i for i, w in enumerate(words) if w in tr_words]
    te_idx = [i for i, w in enumerate(words) if w in te_words]
    
    # Final sanity check to ensure absolute lexical separation
    train_word_set = set(words[i] for i in tr_idx)
    test_word_set = set(words[i] for i in te_idx)
    overlap = train_word_set.intersection(test_word_set)
    
    if len(overlap) > 0:
        print(f"❌ CRITICAL ERROR: Word leakage detected! Overlapping words: {overlap}")
    else:
        print("✅ SUCCESS: Zero word overlap between Train and Test splits. Leakage sealed.")
    print("="*50 + "\n")
        
    return tr_idx, te_idx


def train():
    # Setup execution device (utilizes your RTX 3050 if local drivers match, or falls back)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Computing Device detected: {device}")
    
    # Initialize Dataset
    dataset = TESSDataset()
    
    # Ensure standard weights save directory exists (navigates out of speech_pipeline/)
    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)

    # Calculate Lexical Splits
    tr_idx, te_idx = get_lexical_split(dataset, test_size=0.2)
    print(f"Allocated Dataset Samples -> Train Count: {len(tr_idx)} | Test Count: {len(te_idx)}")

    # Compute class weights using ONLY the training partition labels
    train_labels = [dataset.labels[i] for i in tr_idx]
    classes = np.unique(train_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Calculated Class Weights: {class_weights.cpu().numpy()}\n")

    # Construct DataLoaders
    train_loader = DataLoader(Subset(dataset, tr_idx), batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(Subset(dataset, te_idx), batch_size=BATCH, shuffle=False)

    # Initialize Network, Optimizer, and Loss Criteria
    model = SpeechEmotionModel(INPUT, HIDDEN, CLASSES).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = 0.0
    
    # Core Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            
            running_loss += loss.item()

        # Evaluation Step per Epoch
        model.eval()
        correct = total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                
                correct += (pred == y).sum().item()
                total += y.size(0)

        epoch_acc = (100 * correct) / total
        avg_loss = running_loss / len(train_loader)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Test Acc: {epoch_acc:.2f}%")

        # Save model checkpoint if performance beats previous records
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            
            # Save weights
            torch.save(model.state_dict(), os.path.join(models_dir, "speech_model_best.pth"))
            # Save frozen test split indices so test.py evaluates on the exact same unseen words
            torch.save(te_idx, os.path.join(models_dir, "test_indices.pt"))
            print(f"   ⭐ New best model saved with accuracy: {best_acc:.2f}%")

    print(f"\n🏁 Training complete. Best Model Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()
