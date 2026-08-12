from __future__ import annotations

import random
from pathlib import Path

import soundfile as sf
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.ast_features import (
    ASTBinaryClassifier,
    ASTFeatureExtractor,
)
from app.ml.dataset import discover_samples


SEED = 42
DATA_ROOT = "data/raw/LA"

MAX_PER_CLASS = 200
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3

EMBEDDING_PATH = Path(
    "models/ast_embeddings.pt"
)

CLASSIFIER_PATH = Path(
    "models/ast_binary_classifier.pt"
)


def set_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)


def load_balanced_samples():
    samples = discover_samples(
        DATA_ROOT,
        split="train",
    )

    bonafide = [
        x for x in samples
        if x[1] == 0
    ]

    spoof = [
        x for x in samples
        if x[1] == 1
    ]

    random.shuffle(bonafide)
    random.shuffle(spoof)

    count = min(
        MAX_PER_CLASS,
        len(bonafide),
        len(spoof),
    )

    selected = (
        bonafide[:count]
        + spoof[:count]
    )

    random.shuffle(selected)

    print(
        f"Selected samples: {len(selected)}"
    )
    print(
        f"Bonafide: {sum(y == 0 for _, y in selected)}"
    )
    print(
        f"Spoof: {sum(y == 1 for _, y in selected)}"
    )

    return selected


def extract_embeddings(samples):
    extractor = ASTFeatureExtractor()

    embeddings = []
    labels = []

    total = len(samples)

    for index, (audio_path, label) in enumerate(
        samples,
        start=1,
    ):
        audio, sample_rate = sf.read(
            audio_path,
            dtype="float32",
        )

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        embedding = extractor.extract(
            audio,
            sample_rate,
        )

        embeddings.append(
            embedding.squeeze(0)
        )

        labels.append(label)

        if index % 25 == 0 or index == total:
            print(
                f"Embedding {index}/{total}"
            )

    X = torch.stack(embeddings)
    y = torch.tensor(
        labels,
        dtype=torch.long,
    )

    return X, y


def main():
    set_seed()

    samples = load_balanced_samples()

    X, y = extract_embeddings(
        samples
    )

    print(
        f"Embedding tensor: {X.shape}"
    )
    print(
        f"Label tensor: {y.shape}"
    )

    EMBEDDING_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "embeddings": X,
            "labels": y,
        },
        EMBEDDING_PATH,
    )

    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=SEED,
            stratify=y,
        )
    )

    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=SEED,
            stratify=y_temp,
        )
    )

    train_loader = DataLoader(
        TensorDataset(
            X_train,
            y_train,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            X_val,
            y_val,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = ASTBinaryClassifier()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()

        correct = 0
        total = 0
        running_loss = 0.0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predictions = (
                logits.argmax(dim=1)
            )

            total += y_batch.size(0)

            correct += (
                predictions == y_batch
            ).sum().item()

        train_accuracy = (
            correct / total
        )

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:

                logits = model(
                    X_batch
                )

                predictions = (
                    logits.argmax(dim=1)
                )

                val_total += y_batch.size(0)

                val_correct += (
                    predictions == y_batch
                ).sum().item()

        val_accuracy = (
            val_correct / val_total
        )

        average_loss = (
            running_loss
            / len(train_loader)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "input_dim": 768,
                    "num_classes": 2,
                },
                CLASSIFIER_PATH,
            )

            print(
                f"Saved best classifier -> "
                f"{CLASSIFIER_PATH}"
            )

    print(
        "\nAST classifier training complete."
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Embeddings saved: "
        f"{EMBEDDING_PATH}"
    )

    print(
        f"Classifier saved: "
        f"{CLASSIFIER_PATH}"
    )

    torch.save(
        {
            "X_test": X_test,
            "y_test": y_test,
        },
        "models/ast_test_split.pt",
    )


if __name__ == "__main__":
    main()