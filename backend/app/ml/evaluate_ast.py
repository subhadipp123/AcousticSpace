from __future__ import annotations

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from app.ml.ast_features import ASTBinaryClassifier


MODEL_PATH = "models/ast_binary_classifier.pt"
TEST_PATH = "models/ast_test_split.pt"


def main():
    print("Loading AST test split...")

    data = torch.load(
        TEST_PATH,
        map_location="cpu",
    )

    X_test = data["X_test"]
    y_test = data["y_test"]

    print("Test embeddings:", X_test.shape)
    print("Test labels:", y_test.shape)

    model = ASTBinaryClassifier()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():
        logits = model(X_test)

        predictions = logits.argmax(
            dim=1
        )

    y_true = y_test.tolist()
    y_pred = predictions.tolist()

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

    print("\n===== AST TEST RESULTS =====")
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