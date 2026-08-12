from __future__ import annotations

from pathlib import Path

import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)


MODEL_NAME = (
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

AST_OUTPUT_DIR = Path(
    "models/ast_asvspoof"
)


LABEL2ID = {
    "bonafide": 0,
    "spoof": 1,
}

ID2LABEL = {
    0: "bonafide",
    1: "spoof",
}


def load_ast_for_asvspoof():
    """
    Load the pretrained Audio Spectrogram Transformer
    and replace its AudioSet classifier with a
    binary bonafide/spoof classifier.
    """

    print(
        f"Loading pretrained AST: {MODEL_NAME}"
    )

    extractor = (
        AutoFeatureExtractor.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        AutoModelForAudioClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            ignore_mismatched_sizes=True,
        )
    )

    return extractor, model


def save_ast_model(
    extractor,
    model,
):
    AST_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    extractor.save_pretrained(
        AST_OUTPUT_DIR
    )

    model.save_pretrained(
        AST_OUTPUT_DIR
    )

    print(
        f"AST model saved to: "
        f"{AST_OUTPUT_DIR}"
    )