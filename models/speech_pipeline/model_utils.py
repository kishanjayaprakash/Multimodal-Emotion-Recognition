import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Absolute path based on your exact Drive structure shown in the image
DATA_DIR = "/content/drive/MyDrive/Multimodal Emotion Recognition/data/TESS"

class TESSDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, sr=16000, max_duration=3.0, n_mfcc=40):
        self.data_dir = data_dir
        self.sr = sr
        self.max_len = int(sr * max_duration)
        self.n_mfcc = n_mfcc

        self.emotion_map = {
            'angry': 0, 'disgust': 1, 'fear': 2,
            'happy': 3, 'neutral': 4, 'ps': 5, 'sad': 6
        }

        self.file_paths = []
        self.labels = []

        # Crawl the TESS folder structure
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(".wav"):
                    # TESS format is typically Speaker_Word_Emotion.wav
                    parts = f.replace(".wav", "").split("_")
                    if len(parts) >= 3:
                        emo = parts[-1].lower().strip()
                        if emo in self.emotion_map:
                            self.file_paths.append(os.path.join(root, f))
                            self.labels.append(self.emotion_map[emo])

        print(f"\n[Data Setup] Successfully loaded {len(self.file_paths)} audio paths.")
        if len(self.file_paths) == 0:
            raise ValueError(f"❌ Zero audio files found. Double-check path: {data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        # Preprocessing: Load and trim silence [cite: 35]
        audio, _ = librosa.load(path, sr=self.sr)
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # Standardize length (padding / truncation) [cite: 35]
        if len(audio) > self.max_len:
            audio = audio[:self.max_len]
        else:
            audio = np.pad(audio, (0, self.max_len - len(audio)))

        # Feature Extraction: MFCC + Deltas [cite: 38]
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        d1 = librosa.feature.delta(mfcc, axis=1)
        d2 = librosa.feature.delta(mfcc, order=2, axis=1)

        feat = np.concatenate([mfcc, d1, d2], axis=0).T
        
        # Z-score Normalization per audio file
        feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)

        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label)


class AttentionLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden, 1)

    def forward(self, x):
        score = self.attn(x)
        weight = torch.softmax(score, dim=1)
        context = (weight * x).sum(dim=1)
        return context, weight


class SpeechEmotionModel(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super().__init__()

        # Temporal Modelling Block [cite: 12, 41]
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Contextual Modelling Block (Attention mechanism) [cite: 13, 42]
        self.attention = AttentionLayer(hidden * 2)

        # Classifier Block [cite: 7, 45]
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x, return_features=False):
        lstm_out, _ = self.lstm(x)
        context, attn = self.attention(lstm_out)
        out = self.classifier(context)

        if return_features:
            return out, lstm_out, context, attn
        return out
