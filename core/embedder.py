from __future__ import annotations
import logging
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps sentence-transformers BGE model.

    BGE models require a prefix for query embeddings to align query
    and document embedding spaces. Documents are embedded as-is.
    """

    def __init__(self, config: dict):
        cfg = config["models"]["embedding"]
        self.model_name: str = cfg["name"]
        self.device: str = cfg.get("device", "cpu")
        self.batch_size: int = cfg.get("batch_size", 32)
        self.query_prefix: str = cfg.get(
            "query_prefix",
            "Represent this sentence for searching relevant passages: "
        )

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dim: int = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model ready — dim={self.dim}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string with the BGE query prefix."""
        prefixed = self.query_prefix + query
        return self._encode([prefixed])[0]

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of document chunks (no prefix)."""
        return self._encode(texts)

    def embed(self, text: Union[str, List[str]], is_query: bool = False) -> np.ndarray:
        """Generic embed — prefers the specific methods above."""
        if isinstance(text, str):
            return self.embed_query(text) if is_query else self._encode([text])[0]
        return self.embed_query(text[0]) if is_query else self.embed_documents(text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
        )
