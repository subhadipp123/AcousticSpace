from __future__ import annotations

import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from app.ml.dataset import ASVspoofDataset, discover_samples
from app.ml.model import AudioCNN


SEED = 42
DATA_ROOT = "data/raw/LA"
MODEL_PATH = Path("models/baseline_cnn.pt")

BATCH_SIZE = 8
SAMPLE_RATE = 16000
DURATION = 4.0
MAX_PER_CLASS = 1000


def limit_balanced_samples(samples):
    import random

    random.seed(SEED)

    bonafide = [x for x in samples if x[1] == 0]
    spoof = [x for x in samples if x[1] == 1]

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
    device = torch.device("cpu")

    print(f"Device: {device}")

    samples = discover_samples(
        DATA_ROOT,
        split="train",
    )

    samples = limit_balanced_samples(samples)

    labels = [y for _, y in samples]

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )

    _, test_samples = train_test_split(
        temp_samples,
        test_size=0.50,
        random_state=SEED,
        stratify=[y for _, y in temp_samples],
    )

    print(f"Test samples: {len(test_samples)}")

    test_dataset = ASVspoofDataset(
        test_samples,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = AudioCNN(num_classes=2)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    y_true = []
    y_pred = []

    print("Running test evaluation...")

    with torch.no_grad():
        for waveforms, labels_batch in test_loader:
            outputs = model(waveforms)

            predictions = outputs.argmax(
                dim=1
            )

            y_true.extend(
                labels_batch.tolist()
            )

            y_pred.extend(
                predictions.tolist()
            )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print("\n===== BASELINE RESULTS =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "bonafide",
                "spoof",
            ],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()