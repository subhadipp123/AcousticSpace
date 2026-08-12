from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import soundfile as sf
import torch
from torch.utils.data import Dataset


class ASVspoofDataset(Dataset):
    """
    ASVspoof 2019 Logical Access (LA) dataset.

    Labels:
        0 = bonafide
        1 = spoof
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        sample_rate: int = 16000,
        duration: float = 4.0,
    ):
        self.samples = samples
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        audio_path, label = self.samples[index]

        audio, sample_rate = sf.read(
            audio_path,
            dtype="float32",
        )

        # Convert NumPy audio to PyTorch tensor.
        # SoundFile returns:
        #   [samples] for mono
        #   [samples, channels] for multi-channel
        if audio.ndim == 1:
            waveform = torch.from_numpy(audio).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio.T)

        # Convert multi-channel audio to mono.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

        # Resample if required.
        if sample_rate != self.sample_rate:
            import torchaudio

            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.sample_rate,
            )

        # Make every sample exactly the same length.
        if waveform.shape[1] < self.num_samples:
            padding = self.num_samples - waveform.shape[1]

            waveform = torch.nn.functional.pad(
                waveform,
                (0, padding),
            )
        else:
            waveform = waveform[:, : self.num_samples]

        return (
            waveform.squeeze(0),
            torch.tensor(label, dtype=torch.long),
        )


def find_protocol_file(
    root: str,
    split: str,
) -> Path:
    """
    Locate the ASVspoof 2019 LA countermeasure protocol.
    """

    root_path = Path(root)

    if split == "train":
        protocol_name = "ASVspoof2019.LA.cm.train.trn.txt"
    elif split == "dev":
        protocol_name = "ASVspoof2019.LA.cm.dev.trl.txt"
    elif split == "eval":
        protocol_name = "ASVspoof2019.LA.cm.eval.trl.txt"
    else:
        raise ValueError(
            "split must be 'train', 'dev', or 'eval'"
        )

    matches = list(
        root_path.rglob(protocol_name)
    )

    if not matches:
        raise FileNotFoundError(
            f"Protocol file not found: {protocol_name}"
        )

    return matches[0]


def build_audio_index(
    root: str,
) -> dict[str, str]:
    """
    Build one filename -> full path index.

    This avoids scanning 122,000+ files repeatedly.
    """

    root_path = Path(root)

    print("Building audio index...")

    audio_index: dict[str, str] = {}

    for audio_file in root_path.rglob("*.flac"):
        audio_index[audio_file.stem] = str(
            audio_file
        )

    print(
        f"Indexed audio files: {len(audio_index)}"
    )

    return audio_index


def discover_samples(
    root: str,
    split: str = "train",
) -> List[Tuple[str, int]]:
    """
    Read the ASVspoof 2019 LA CM protocol and
    create (audio_path, label) pairs.

    Protocol format:

        speaker_id file_id codec system label

    Example:

        LA_0079 LA_T_1138215 - - bonafide
        LA_0039 LA_E_2834763 - A11 spoof
    """

    protocol_file = find_protocol_file(
        root,
        split,
    )

    print(
        f"Protocol: {protocol_file.name}"
    )

    audio_index = build_audio_index(root)

    samples: List[Tuple[str, int]] = []

    with protocol_file.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        for line in file:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            file_id = parts[1]
            class_label = parts[-1].lower()

            if class_label == "bonafide":
                label = 0
            elif class_label == "spoof":
                label = 1
            else:
                continue

            audio_path = audio_index.get(
                file_id
            )

            if audio_path is not None:
                samples.append(
                    (
                        audio_path,
                        label,
                    )
                )

    return samples