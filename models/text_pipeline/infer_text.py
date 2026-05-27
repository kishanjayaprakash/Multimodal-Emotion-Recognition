import os
import torch
from transformers import BertTokenizer
from model_utils import TextEmotionModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant surprise", "sad"]

print("🔄 Initializing interactive text inference framework...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TextEmotionModel(num_classes=7).to(device)

WEIGHTS_PATH = "./best_text_model.pth"
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print(f"✅ Text weights successfully loaded from: {WEIGHTS_PATH}")
else:
    print(f"⚠️ Warning: Checkpoint not found at '{WEIGHTS_PATH}'. Running on randomized base weights.")

model.eval()

print("\n" + "="*60)
print("             TEXT MODALITY LIVE INFERENCE CONSOLE")
print("="*60)
print("Type 'exit' or 'quit' to terminate the live session.\n")

while True:
    user_input = input("Enter a text string to analyze ➔ ")
    if user_input.lower() in ['exit', 'quit']:
        print("👋 Terminating text inference session.")
        break
    if not user_input.strip():
        continue
        
    formatted_text = f"The word is {user_input.lower().strip()}"
    
    encoding = tokenizer(
        formatted_text,
        padding='max_length',
        truncation=True,
        max_length=16,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        prediction_idx = torch.argmax(probabilities).item()
        
    print(f"🔮 Predicted Emotion: {EMOTIONS[prediction_idx].upper()} ({probabilities[prediction_idx]*100:.2f}% Confidence)")
    print("-" * 60)
