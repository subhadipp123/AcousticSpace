from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch

from app.ml.model import AudioCNN
from app.ml.ast_inference import predict_ast_audio


CNN_MODEL_PATH = Path("models/baseline_cnn.pt")

_cnn_model: AudioCNN | None = None


def load_cnn_model() -> AudioCNN:
    global _cnn_model

    if _cnn_model is not None:
        return _cnn_model

    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CNN model not found: {CNN_MODEL_PATH}"
        )

    model = AudioCNN(num_classes=2)

    checkpoint = torch.load(
        CNN_MODEL_PATH,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    _cnn_model = model

    return _cnn_model


def prepare_cnn_audio(
    audio_path: str,
) -> torch.Tensor:
    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
    )

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    waveform = torch.from_numpy(audio)

    if sample_rate != 16000:
        import torchaudio

        waveform = waveform.unsqueeze(0)

        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            16000,
        )

        waveform = waveform.squeeze(0)

    num_samples = 16000 * 4

    if waveform.shape[0] < num_samples:
        waveform = torch.nn.functional.pad(
            waveform,
            (0, num_samples - waveform.shape[0]),
        )
    else:
        waveform = waveform[:num_samples]

    return waveform


def predict_cnn_audio(
    audio_path: str,
) -> dict:
    model = load_cnn_model()

    waveform = prepare_cnn_audio(
        audio_path
    )

    waveform = waveform.unsqueeze(0)

    with torch.no_grad():
        logits = model(waveform)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    predicted_class = int(
        probabilities.argmax().item()
    )

    confidence = float(
        probabilities[predicted_class].item()
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
        "model": "CNN",
    }


def predict_audio(
    audio_path: str,
) -> dict:
    """
    Run both CNN and AST predictions.

    CNN is the baseline.
    AST is the stronger experimental model.
    """

    cnn_result = predict_cnn_audio(
        audio_path
    )

    ast_result = predict_ast_audio(
        audio_path
    )

    return {
        "cnn": cnn_result,
        "ast": ast_result,
        "primary": ast_result,
    }