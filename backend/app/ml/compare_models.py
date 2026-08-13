from __future__ import annotations

import random

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from app.ml.dataset import (
    discover_samples,
)

from app.ml.inference import (
    predict_cnn_audio,
)

from app.ml.ast_inference import (
    predict_ast_audio,
)


SEED = 42

SAMPLES_PER_CLASS = 50

DATA_ROOT = "data/raw/LA"


def main():
    random.seed(SEED)

    samples = discover_samples(
        DATA_ROOT,
        split="dev",
    )

    bonafide = [
        item
        for item in samples
        if item[1] == 0
    ]

    spoof = [
        item
        for item in samples
        if item[1] == 1
    ]

    random.shuffle(
        bonafide
    )

    random.shuffle(
        spoof
    )

    evaluation_samples = (
        bonafide[:SAMPLES_PER_CLASS]
        + spoof[:SAMPLES_PER_CLASS]
    )

    random.shuffle(
        evaluation_samples
    )

    print(
        "Common development-set "
        f"samples: "
        f"{len(evaluation_samples)}"
    )

    cnn_true = []
    cnn_pred = []

    ast_true = []
    ast_pred = []

    for index, (
        audio_path,
        label,
    ) in enumerate(
        evaluation_samples,
        start=1,
    ):
        cnn_result = (
            predict_cnn_audio(
                audio_path
            )
        )

        ast_result = (
            predict_ast_audio(
                audio_path
            )
        )

        cnn_true.append(label)

        cnn_pred.append(
            cnn_result["class_id"]
        )

        ast_true.append(label)

        ast_pred.append(
            ast_result["class_id"]
        )

        if (
            index % 10 == 0
            or index
            == len(
                evaluation_samples
            )
        ):
            print(
                f"Processed "
                f"{index}/"
                f"{len(evaluation_samples)}"
            )


    def print_metrics(
        model_name: str,
        y_true: list[int],
        y_pred: list[int],
    ):
        print(
            f"\n===== "
            f"{model_name} ====="
        )

        print(
            "Accuracy : "
            f"{accuracy_score(y_true, y_pred):.4f}"
        )

        print(
            "Precision: "
            f"{precision_score(y_true, y_pred, zero_division=0):.4f}"
        )

        print(
            "Recall   : "
            f"{recall_score(y_true, y_pred, zero_division=0):.4f}"
        )

        print(
            "F1 Score : "
            f"{f1_score(y_true, y_pred, zero_division=0):.4f}"
        )

        print(
            "Confusion Matrix:"
        )

        print(
            confusion_matrix(
                y_true,
                y_pred,
            )
        )


    print_metrics(
        "CNN BASELINE",
        cnn_true,
        cnn_pred,
    )

    print_metrics(
        "AST MODEL",
        ast_true,
        ast_pred,
    )


if __name__ == "__main__":
    main()