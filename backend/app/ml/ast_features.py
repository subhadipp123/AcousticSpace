from __future__ import annotations

import torch
import torch.nn as nn

from app.ml.ast_model import load_ast_for_asvspoof


class ASTFeatureExtractor:
    def __init__(self):
        self.extractor, self.model = (
            load_ast_for_asvspoof()
        )

        self.model.eval()

        # Freeze all AST parameters.
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def extract(
        self,
        waveform,
        sample_rate: int = 16000,
    ) -> torch.Tensor:

        inputs = self.extractor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        with torch.no_grad():
            hidden_states = (
                self.model.audio_spectrogram_transformer(
                    **inputs
                ).last_hidden_state
            )

        # Mean-pool transformer tokens.
        embedding = hidden_states.mean(
            dim=1
        )

        return embedding


class ASTBinaryClassifier(nn.Module):
    """
    Small classifier operating on frozen AST embeddings.
    """

    def __init__(
        self,
        input_dim: int = 768,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(
                input_dim,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(
                256,
                2,
            ),
        )

    def forward(self, x):
        return self.classifier(x)