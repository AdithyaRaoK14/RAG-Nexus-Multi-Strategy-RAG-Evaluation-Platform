from __future__ import annotations
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

from core.schema import RAGResponse, RAGTrace
from core.retriever import Retriever
from core.reranker import Reranker
from core.generator import Generator

logger = logging.getLogger(__name__)


class BaseRAGStrategy(ABC):
    """
    Contract every RAG strategy must satisfy.

    Subclasses override `retrieve_and_generate`. The base class wraps it
    with timing and trace finalisation so strategies don't need to worry
    about boilerplate.
    """

    def __init__(
        self,
        config: dict,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
    ):
        self.config = config
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.cfg_ret = config["retrieval"]
        self.strategy_name: str = self.__class__.__name__

    def run(
        self,
        query: str,
        domain_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Public entrypoint. Times the strategy and ensures trace is complete.
        """
        t0 = time.perf_counter()
        trace = RAGTrace(query=query, strategy=self.strategy_name)

        response = self.retrieve_and_generate(
            query=query,
            trace=trace,
            domain_filter=domain_filter,
            top_k=top_k or self.cfg_ret["top_k"],
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        response.latency_ms = latency_ms
        response.trace.latency_ms = latency_ms
        return response

    @abstractmethod
    def retrieve_and_generate(
        self,
        query: str,
        trace: RAGTrace,
        domain_filter: Optional[str],
        top_k: int,
    ) -> RAGResponse:
        """
        Core strategy logic. Must return a RAGResponse.
        The trace object is passed in and should be populated.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers available to all strategies
    # ------------------------------------------------------------------

    def _score_confidence(self, query: str, chunks: list) -> float:
        """
        Estimate retrieval confidence from top reranker score.
        Cheap proxy that avoids an extra LLM call.
        """
        if not chunks:
            return 0.0
        top_score = chunks[0].final_score
        return round(float(top_score), 4)

    def _format_context(self, chunks: list) -> list[str]:
        return [c.text for c in chunks]

    def __repr__(self) -> str:
        return f"<{self.strategy_name}>"
