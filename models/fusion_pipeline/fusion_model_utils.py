
import os
import librosa
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset

from transformers import BertTokenizer
from transformers import BertModel


emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "ps": 5,
    "sad": 6
}


class FusionDataset(Dataset):

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

                    path = os.path.join(root, file)

                    text = f"The word is {word}"

                    label = emotion_map[emotion]

                    self.samples.append(
                        (speaker, path, text, label)
                    )

    def extract_mfcc(self, path):

        signal, sr = librosa.load(
            path,
            sr=22050
        )

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=40
        )

        mfcc = np.mean(mfcc.T, axis=0)

        return torch.tensor(
            mfcc,
            dtype=torch.float
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        speaker, path, text, label = self.samples[idx]

        speech_features = self.extract_mfcc(path)

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=16,
            return_tensors='pt'
        )

        input_ids = encoding[
            'input_ids'
        ].squeeze(0)

        attention_mask = encoding[
            'attention_mask'
        ].squeeze(0)

        return (
            speech_features,
            input_ids,
            attention_mask,
            torch.tensor(label)
        )


class FusionModel(nn.Module):

    def __init__(self):

        super(FusionModel, self).__init__()

        # Speech branch
        self.speech_fc = nn.Linear(40, 128)

        # Text branch
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        # Fusion layers
        self.fc1 = nn.Linear(
            128 + 768,
            256
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(
            256,
            7
        )

    def forward(
        self,
        speech_features,
        input_ids,
        attention_mask
    ):

        # Speech embedding
        speech_x = self.speech_fc(
            speech_features
        )

        # Text embedding
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_x = bert_output.pooler_output

        # Concatenate
        combined = torch.cat(
            (speech_x, text_x),
            dim=1
        )

        x = self.fc1(combined)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x
