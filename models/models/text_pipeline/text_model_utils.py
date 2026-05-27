
import os
import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer
from transformers import BertModel

import torch.nn as nn


emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "ps": 5,
    "sad": 6
}


class TESSTextDataset(Dataset):

    def __init__(self, data_dir):

        self.samples = []

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        for root, dirs, files in os.walk(data_dir):

            for file in files:

                if file.endswith(".wav"):

                    parts = file.replace(".wav", "").split("_")

                    if len(parts) < 3:
                        continue

                    speaker = parts[0]
                    word = parts[1].lower()
                    emotion = parts[2].lower()

                    if emotion not in emotion_map:
                        continue

                    text = f"The word is {word}"

                    label = emotion_map[emotion]

                    full_path = os.path.join(root, file)

                    self.samples.append(
                        (speaker, full_path, text, label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        _, _, text, label = self.samples[idx]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=16,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)

        attention_mask = encoding[
            'attention_mask'
        ].squeeze(0)

        return (
            input_ids,
            attention_mask,
            torch.tensor(label)
        )


class TextEmotionModel(nn.Module):

    def __init__(self, num_classes):

        super(TextEmotionModel, self).__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(768, num_classes)

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output

        x = self.dropout(pooled_output)

        x = self.fc(x)

        return x
