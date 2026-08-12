from __future__ import annotations

from typing import List

import soundfile as sf
import torch

from app.ml.inference import load_cnn_model


SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5

WINDOW_SAMPLES = int(
    SAMPLE_RATE * WINDOW_SECONDS
)

HOP_SAMPLES = int(
    SAMPLE_RATE * HOP_SECONDS
)


def load_audio(
    audio_path: str,
) -> torch.Tensor:
    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
    )

    if audio.ndim == 1:
        waveform = torch.from_numpy(audio)
    else:
        waveform = torch.from_numpy(
            audio.mean(axis=1)
        )

    if sample_rate != SAMPLE_RATE:
        import torchaudio

        waveform = waveform.unsqueeze(0)

        waveform = (
            torchaudio.functional.resample(
                waveform,
                sample_rate,
                SAMPLE_RATE,
            )
        )

        waveform = waveform.squeeze(0)

    return waveform


def analyze_segments(
    audio_path: str,
) -> List[dict]:
    """
    Analyze overlapping 1-second audio
    segments using the trained CNN baseline.
    """

    model = load_cnn_model()

    waveform = load_audio(
        audio_path
    )

    total_samples = waveform.shape[0]

    segments = []

    start = 0

    while start < total_samples:
        end = (
            start + WINDOW_SAMPLES
        )

        segment = waveform[start:end]

        actual_end = min(
            end,
            total_samples,
        )

        if segment.shape[0] < WINDOW_SAMPLES:
            segment = torch.nn.functional.pad(
                segment,
                (
                    0,
                    WINDOW_SAMPLES
                    - segment.shape[0],
                ),
            )

        segment = segment.unsqueeze(0)

        with torch.no_grad():
            logits = model(
                segment
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]

        bonafide_probability = float(
            probabilities[0].item()
        )

        spoof_probability = float(
            probabilities[1].item()
        )

        predicted_class = int(
            probabilities.argmax().item()
        )

        confidence = float(
            probabilities[
                predicted_class
            ].item()
        )

        segments.append(
            {
                "start_seconds": round(
                    start / SAMPLE_RATE,
                    3,
                ),
                "end_seconds": round(
                    actual_end / SAMPLE_RATE,
                    3,
                ),
                "prediction": (
                    "bonafide"
                    if predicted_class == 0
                    else "spoof"
                ),
                "confidence": confidence,
                "bonafide_probability":
                    bonafide_probability,
                "spoof_probability":
                    spoof_probability,
                "suspicious":
                    spoof_probability >= 0.70,
            }
        )

        start += HOP_SAMPLES

    return segments