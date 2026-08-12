"""
Audio preprocessing pipeline for AcousticSpace.

Extracts two things from an audio clip:
1. Mel-spectrogram - the standard representation of speech content
2. RIR-proxy features - simplified signal characteristics that correlate
   with room acoustics (true blind RIR estimation needs specialized methods;
   these proxies are appropriate for a baseline classifier).
"""

import librosa
import numpy as np


def load_audio(file_path: str, sr: int = 16000):
    """Load an audio file and resample to a fixed sample rate."""
    audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    return audio, sample_rate


def extract_mel_spectrogram(audio: np.ndarray, sr: int, n_mels: int = 128):
    """Standard mel-spectrogram - captures the vocal/speech content."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def extract_rir_features(audio: np.ndarray, sr: int) -> dict:
    """
    Simplified proxy features for background acoustic reflection:
    - spectral_centroid: brightness of the sound, and how it varies
    - spectral_flatness: reverberant tails sound more "noise-like"
    - rms_energy_std: rooms smooth out loudness variation over time
    """
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
    rms = librosa.feature.rms(y=audio)[0]

    return {
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "spectral_flatness_mean": float(np.mean(spectral_flatness)),
        "rms_energy_std": float(np.std(rms)),
    }


def preprocess_audio(file_path: str) -> dict:
    """Full pipeline: load audio, extract spectrogram + RIR-proxy features."""
    audio, sr = load_audio(file_path)
    return {
        "mel_spectrogram": extract_mel_spectrogram(audio, sr),
        "rir_features": extract_rir_features(audio, sr),
        "duration_seconds": len(audio) / sr,
    }
    
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def generate_spectrogram(
    audio_path: str,
    output_dir: str = "data/generated",
) -> str:
    """
    Generate a mel-spectrogram image for an audio file.
    Returns the relative path to the generated image.
    """

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_file = Path(audio_path)

    y, sr = librosa.load(
        str(audio_file),
        sr=16000,
        mono=True,
    )

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=1024,
        hop_length=256,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    image_name = (
        f"{audio_file.stem}_mel_spectrogram.png"
    )

    image_path = output_path / image_name

    plt.figure(
        figsize=(10, 4),
    )

    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=256,
        x_axis="time",
        y_axis="mel",
    )

    plt.colorbar(
        format="%+2.0f dB",
    )

    plt.title("AcousticSpace Mel Spectrogram")
    plt.tight_layout()

    plt.savefig(
        image_path,
        dpi=150,
    )

    plt.close()

    return str(image_path)