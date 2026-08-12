from __future__ import annotations

from pathlib import Path
import random

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from app.ml.dataset import ASVspoofDataset, discover_samples
from app.ml.model import AudioCNN


SEED = 42

DATA_ROOT = "data/raw/LA"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "baseline_cnn.pt"

# CPU-friendly first baseline
MAX_PER_CLASS = 1000
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

SAMPLE_RATE = 16000
DURATION = 4.0


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def limit_balanced_samples(samples):
    bonafide = [item for item in samples if item[1] == 0]
    spoof = [item for item in samples if item[1] == 1]

    random.shuffle(bonafide)
    random.shuffle(spoof)

    count = min(
        MAX_PER_CLASS,
        len(bonafide),
        len(spoof),
    )

    selected = bonafide[:count] + spoof[:count]

    random.shuffle(selected)

    return selected


def main():
    set_seed(SEED)

    device = torch.device("cpu")
    print(f"Device: {device}")

    all_samples = discover_samples(
        DATA_ROOT,
        split="train",
    )

    print(f"Available labelled samples: {len(all_samples)}")

    samples = limit_balanced_samples(all_samples)

    print(f"Samples selected for baseline: {len(samples)}")
    print(
        "Selected bonafide:",
        sum(1 for _, y in samples if y == 0),
    )
    print(
        "Selected spoof:",
        sum(1 for _, y in samples if y == 1),
    )

    labels = [y for _, y in samples]

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.50,
        random_state=SEED,
        stratify=[y for _, y in temp_samples],
    )

    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    train_dataset = ASVspoofDataset(
        train_samples,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
    )

    val_dataset = ASVspoofDataset(
        val_samples,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = AudioCNN(
        num_classes=2
    ).to(device)

    # Balanced class weights.
    train_bonafide = sum(
        1 for _, y in train_samples if y == 0
    )
    train_spoof = sum(
        1 for _, y in train_samples if y == 1
    )

    total_train = train_bonafide + train_spoof

    class_weights = torch.tensor(
        [
            total_train / (2 * train_bonafide),
            total_train / (2 * train_spoof),
        ],
        dtype=torch.float32,
        device=device,
    )

    print(
        "Class weights:",
        class_weights.tolist(),
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for batch_index, (waveforms, labels_batch) in enumerate(
            train_loader,
            start=1,
        ):
            waveforms = waveforms.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()

            outputs = model(waveforms)

            loss = criterion(
                outputs,
                labels_batch,
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            total += labels_batch.size(0)

            correct += (
                predictions == labels_batch
            ).sum().item()

            if batch_index % 25 == 0:
                print(
                    f"  Batch {batch_index}/{len(train_loader)}"
                )

        train_accuracy = (
            correct / total
            if total
            else 0.0
        )

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for waveforms, labels_batch in val_loader:
                waveforms = waveforms.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(waveforms)

                predictions = outputs.argmax(
                    dim=1
                )

                val_total += labels_batch.size(0)

                val_correct += (
                    predictions == labels_batch
                ).sum().item()

        val_accuracy = (
            val_correct / val_total
            if val_total
            else 0.0
        )

        average_loss = (
            running_loss / len(train_loader)
        )

        print(
            f"Loss: {average_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "sample_rate": SAMPLE_RATE,
                    "duration": DURATION,
                    "num_classes": 2,
                },
                MODEL_PATH,
            )

            print(
                f"Saved best model -> {MODEL_PATH}"
            )

    print("\nTraining complete.")
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )
    print(
        f"Model saved at: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()