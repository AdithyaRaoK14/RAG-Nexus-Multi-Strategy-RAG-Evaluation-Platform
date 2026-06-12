from __future__ import annotations
from typing import Optional

from strategies.base import BaseRAGStrategy
from core.schema import RAGResponse, RAGTrace


class NaiveRAG(BaseRAGStrategy):
    """
    Baseline strategy: dense vector search → rerank → generate.

    No query transformation. No sparse retrieval. No graph.
    This is the anchor every other strategy is benchmarked against.
    """

    def retrieve_and_generate(
        self,
        query: str,
        trace: RAGTrace,
        domain_filter: Optional[str],
        top_k: int,
    ) -> RAGResponse:

        # 1. Dense retrieval
        candidates = self.retriever.dense_search(
            query=query,
            top_k=top_k,
            domain_filter=domain_filter,
        )
        trace.dense_scores = [round(c.dense_score, 4) for c in candidates]

        # 2. Rerank
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
            strategy="naive_rag",
            trace=trace,
            latency_ms=0.0,    # filled by base.run()
            confidence=confidence,
        )
