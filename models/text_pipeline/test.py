
import torch

from transformers import BertTokenizer

from model_utils import TextEmotionModel


MODEL_PATH = "/content/drive/MyDrive/Multimodal Emotion Recognition/models/text_pipeline/best_text_model.pth"

NUM_CLASSES = 7


emotion_labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "pleasant_surprise",
    6: "sad"
}


device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

model = TextEmotionModel(NUM_CLASSES).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()


def predict_emotion(text):

    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=16,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():

        outputs = model(
            input_ids,
            attention_mask
        )

        prediction = torch.argmax(outputs, dim=1).item()

    return emotion_labels[prediction]


if __name__ == "__main__":

    text = input("Enter text: ")

    emotion = predict_emotion(text)

    print(f"Predicted Emotion: {emotion}")
