from __future__ import annotations
from typing import Optional, List

from strategies.base import BaseRAGStrategy
from core.schema import RAGResponse, RAGTrace, RetrievedChunk


class AdvancedRAG(BaseRAGStrategy):
    """
    Advanced RAG with pre-retrieval and post-retrieval enhancements.

    Pre-retrieval:
      1. Query rewriting — makes ambiguous queries more retrieval-friendly
      2. HyDE — generates a hypothetical document, embeds it, retrieves
         using that embedding. Both signals merged via RRF.

    Post-retrieval:
      3. Contextual compression — drops reranked chunks below score threshold
      4. Generation from compressed, high-confidence context
    """

    COMPRESS_THRESHOLD = 0.25

    def retrieve_and_generate(
        self,
        query: str,
        trace: RAGTrace,
        domain_filter: Optional[str],
        top_k: int,
    ) -> RAGResponse:

        # ---- Pre-retrieval -----------------------------------------------

        # 1. Query rewriting
        rewrite_prompt = self.generator.build_rewrite_prompt(query)
        rewritten_query, _ = self.generator.generate(rewrite_prompt)
        rewritten_query = rewritten_query.strip().strip('"').strip("'") or query
        trace.rewrite_query = rewritten_query

        # 2. HyDE
        hyde_prompt = self.generator.build_hyde_prompt(query)
        hyde_doc, _ = self.generator.generate(hyde_prompt)
        trace.hyde_doc = hyde_doc[:300]

        # ---- Retrieval (two signals) -------------------------------------

        # Signal A: hybrid search on rewritten query
        candidates_a = self.retriever.hybrid_search(
            query=rewritten_query,
            top_k=top_k,
            domain_filter=domain_filter,
        )

        # Signal B: dense search on HyDE embedding
        hyde_embedding = self.retriever.embedder.embed_documents([hyde_doc])[0]
        hyde_results_raw = self.retriever.qdrant.search(
            collection_name=self.retriever.collection,
            query_vector=hyde_embedding.tolist(),
            limit=top_k,
            with_payload=True,
        )
        candidates_b: List[RetrievedChunk] = [
            self.retriever._scored_point_to_chunk(r) for r in hyde_results_raw
        ]

        # Merge via RRF
        all_candidates = {c.chunk_id: c for c in candidates_a + candidates_b}
        fused = self.retriever._rrf_fusion(
            [[c.chunk_id for c in candidates_a], [c.chunk_id for c in candidates_b]]
        )
        merged = sorted(
            all_candidates.values(),
            key=lambda c: fused.get(c.chunk_id, 0),
            reverse=True,
        )[:top_k]

        trace.dense_scores = [round(c.dense_score, 4) for c in merged]
        trace.sparse_scores = [round(c.sparse_score, 4) for c in merged]

        # ---- Post-retrieval ---------------------------------------------

        # 3. Rerank
        reranked = self.reranker.rerank(query, merged)
        trace.reranker_scores = [round(c.rerank_score, 4) for c in reranked if c.rerank_score]

        # 4. Contextual compression
        chunks = [c for c in reranked if (c.rerank_score or 0) >= self.COMPRESS_THRESHOLD]
        if not chunks:
            chunks = reranked[:2]

        confidence = self._score_confidence(query, chunks)
        trace.confidence = confidence

        context = self._format_context(chunks)
        prompt = self.generator.build_rag_prompt(query, context)
        answer, _ = self.generator.generate(prompt)
        trace.tokens_estimated = self.generator.estimate_tokens(prompt)

        return RAGResponse(
            query=query,
            answer=answer,
            sources=chunks,
            strategy="advanced_rag",
            trace=trace,
            latency_ms=0.0,
            confidence=confidence,
        )
