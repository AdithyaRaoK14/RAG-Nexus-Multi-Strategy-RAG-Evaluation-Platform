from __future__ import annotations
from typing import Optional

from strategies.base import BaseRAGStrategy
from core.schema import RAGResponse, RAGTrace


class HybridRAG(BaseRAGStrategy):
    """
    Production-style hybrid retrieval.

    Pipeline:
      dense (Qdrant cosine) + sparse (BM25Okapi)
        → RRF fusion
        → cross-encoder rerank
        → generate

    BM25 captures exact keyword matches that dense search misses,
    especially for technical terms, acronyms, and proper nouns.
    """

    def retrieve_and_generate(
        self,
        query: str,
        trace: RAGTrace,
        domain_filter: Optional[str],
        top_k: int,
    ) -> RAGResponse:

        # 1. Hybrid retrieval (dense + BM25 + RRF internally)
        candidates = self.retriever.hybrid_search(
            query=query,
            top_k=top_k,
            domain_filter=domain_filter,
        )
        trace.dense_scores = [round(c.dense_score, 4) for c in candidates]
        trace.sparse_scores = [round(c.sparse_score, 4) for c in candidates]

        # 2. Cross-encoder rerank
        chunks = self.reranker.rerank(query, candidates)
        trace.reranker_scores = [round(c.rerank_score, 4) for c in chunks if c.rerank_score]

        # 3. Confidence
        confidence = self._score_confidence(query, chunks)
        trace.confidence = confidence

        # 4. Generate
        context = self._format_context(chunks)
        prompt = self.generator.build_rag_prompt(query, context)
        answer, _ = self.generator.generate(prompt)
        trace.tokens_estimated = self.generator.estimate_tokens(prompt)

        return RAGResponse(
            query=query,
            answer=answer,
            sources=chunks,
            strategy="hybrid_rag",
            trace=trace,
            latency_ms=0.0,
            confidence=confidence,
        )
