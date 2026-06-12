from __future__ import annotations
import logging
import math
from typing import List, Optional

from sentence_transformers import CrossEncoder
from core.schema import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

    Takes query + candidate chunks, returns them sorted by cross-encoder
    score with rerank_score (sigmoid-normalised to [0,1]) populated.
    """

    def __init__(self, config: dict):
        cfg = config["models"]["reranker"]
        self.model_name: str = cfg["name"]
        self.device: str = cfg.get("device", "cpu")
        self.top_k: int = cfg.get("top_k", 5)

        logger.info(f"Loading reranker: {self.model_name}")
        self.model = CrossEncoder(self.model_name, device=self.device)
        logger.info("Reranker ready")

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Rerank chunks by cross-encoder score. Returns top_k sorted descending."""
        if not chunks:
            return []
        k = top_k or self.top_k
        pairs = [(query, chunk.text) for chunk in chunks]
        scores: List[float] = self.model.predict(pairs).tolist()

        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = round(self._sigmoid(score), 4)

        reranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
        return reranked[:k]

    def score_pair(self, query: str, text: str) -> float:
        """Score a single query-document pair. Used for confidence checks."""
        raw = float(self.model.predict([(query, text[:512])]))
        return round(self._sigmoid(raw), 4)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))
