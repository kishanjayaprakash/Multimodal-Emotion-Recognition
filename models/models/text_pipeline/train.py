
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset

from text_model_utils import (
    TESSTextDataset,
    TextEmotionModel
)


DATA_DIR = "/content/drive/MyDrive/Multimodal Emotion Recognition/data/TESS"

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5

NUM_CLASSES = 7

MODEL_SAVE_PATH = "/content/drive/MyDrive/Multimodal Emotion Recognition/models/text_pipeline/best_text_model.pth"


def train():

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Training on device: {device}")

    dataset = TESSTextDataset(DATA_DIR)

    train_indices = []
    val_indices = []

    for idx, sample in enumerate(dataset.samples):

        speaker = sample[0].upper()

        if speaker.startswith("OAF"):
            train_indices.append(idx)

        elif speaker.startswith("YAF"):
            val_indices.append(idx)

    print(f"\nTrain samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = TextEmotionModel(NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0.0

        for input_ids, attention_mask, labels in train_loader:

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids,
                attention_mask
            )

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for input_ids, attention_mask, labels in val_loader:

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(
                    input_ids,
                    attention_mask
                )

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        val_accuracy = 100 * correct / total

        avg_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Loss: {avg_loss:.4f} "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )

            print("✅ Saved new best model!")

    print("\nTraining complete.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")


if __name__ == "__main__":
    train()
