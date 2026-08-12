from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch

from app.ml.ast_features import (
    ASTBinaryClassifier,
    ASTFeatureExtractor,
)


CLASSIFIER_PATH = Path(
    "models/ast_binary_classifier.pt"
)


_ast_extractor: ASTFeatureExtractor | None = None
_ast_classifier: ASTBinaryClassifier | None = None


def load_ast_inference():
    global _ast_extractor
    global _ast_classifier

    if (
        _ast_extractor is not None
        and _ast_classifier is not None
    ):
        return (
            _ast_extractor,
            _ast_classifier,
        )

    _ast_extractor = ASTFeatureExtractor()

    _ast_classifier = ASTBinaryClassifier()

    checkpoint = torch.load(
        CLASSIFIER_PATH,
        map_location="cpu",
    )

    _ast_classifier.load_state_dict(
        checkpoint["model_state_dict"]
    )

    _ast_classifier.eval()

    return (
        _ast_extractor,
        _ast_classifier,
    )


def predict_ast_audio(
    audio_path: str,
) -> dict:

    extractor, classifier = (
        load_ast_inference()
    )

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

    with torch.no_grad():
        logits = classifier(
            embedding
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    predicted_class = int(
        probabilities.argmax().item()
    )

    confidence = float(
        probabilities[
            predicted_class
        ].item()
    )

    return {
        "label": (
            "bonafide"
            if predicted_class == 0
            else "spoof"
        ),
        "class_id": predicted_class,
        "confidence": confidence,
        "bonafide_probability": float(
            probabilities[0].item()
        ),
        "spoof_probability": float(
            probabilities[1].item()
        ),
        "model": "AST",
    }