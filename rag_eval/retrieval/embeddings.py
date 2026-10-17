"""
Embedding wrapper.

Model choice: sentence-transformers/all-MiniLM-L6-v2
See README "Embedding model choice" for the justification. This is the
*only* place the model name would normally be hardcoded — but even here it
is not: the model name is passed in from config/experiment.yaml, so
swapping embedding models never requires touching this file.

Not required, and not imported, by --dry-run mode.
"""
from __future__ import annotations
import numpy as np


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu", normalize: bool = True):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # local import: real runs only
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
