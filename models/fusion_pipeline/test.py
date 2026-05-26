import os
import glob

# 1. Dynamically locate your text_pipeline folder inside Drive
search_text = glob.glob('/content/drive/**/text_pipeline/text_model_utils.py', recursive=True)
if not search_text:
    raise FileNotFoundError("❌ Could not locate text_pipeline folder in your Drive.")

TEXT_PIPELINE_DIR = os.path.dirname(search_text[0])
test_script_path = os.path.join(TEXT_PIPELINE_DIR, "test.py")

# 2. Define the code contents for test.py
test_code_content = """import os
import sys
import torch
from transformers import BertTokenizer

# Append current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from text_model_utils import TextEmotionModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Pleasant Surprise', 'Sad']

print("🔄 Initializing interactive text inference framework...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TextEmotionModel(num_classes=7).to(device)

# Load weights safely
weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_text_model.pth")
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"✅ Text weights loaded successfully from: {weights_path}")
else:
    print(f"⚠️ Warning: Could not find '{weights_path}'. Running inference on randomized initialization base.")

model.eval()

print("\\n" + "="*50)
print("     TEXT MODALITY LIVE TEST INTERFACE")
print("="*50)
print("Type 'exit' or 'quit' to terminate the session.\\n")

while True:
    user_input = input("Enter text string to analyze ➔ ")
    if user_input.lower() in ['exit', 'quit']:
        print("👋 Terminating test pipeline inference.")
        break
    if not user_input.strip():
        continue
        
    # Tokenize input string
    inputs = tokenizer(
        user_input,
        padding='max_length',
        truncation=True,
        max_length=16,
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        max_idx = torch.argmax(probabilities).item()
        
    print(f"🔮 Predicted Emotion Classification : {EMOTIONS[max_idx]} ({probabilities[max_idx]*100:.2f}% Confidence)")
    print("-" * 50)
"""

# 3. Write the file to your drive directory
with open(test_script_path, "w") as f:
    f.write(test_code_content)

print(f"🎉 Success! 'test.py' has been successfully created and saved inside your folder at:")
print(f"📁 {test_script_path}")
