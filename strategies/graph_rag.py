from __future__ import annotations
import logging
from typing import Optional

from strategies.base import BaseRAGStrategy
from core.schema import RAGResponse, RAGTrace
from knowledge_graph.graph_store import KnowledgeGraphStore

logger = logging.getLogger(__name__)


class GraphRAG(BaseRAGStrategy):
    """
    Graph-augmented retrieval.

    Pipeline:
      1. Extract entities mentioned in the query
      2. Expand via KG neighbourhood (BFS, depth=2)
      3. Use expanded entity set to boost dense retrieval
         (retrieve chunks from source documents linked to those entities)
      4. Fall back to hybrid search when graph is sparse
      5. Rerank and generate
    """

    def __init__(self, config, retriever, reranker, generator, kg_store: KnowledgeGraphStore):
        super().__init__(config, retriever, reranker, generator)
        self.kg = kg_store

    def retrieve_and_generate(
        self,
        query: str,
        trace: RAGTrace,
        domain_filter: Optional[str],
        top_k: int,
    ) -> RAGResponse:

        # 1. Find query entities in graph
        query_entities = self.kg.find_entities_in_text(query)
        trace.extra["query_entities"] = query_entities

        # 2. Expand neighbourhood
        expanded: set = set()
        for entity in query_entities:
            expanded |= self.kg.get_neighbors(entity, depth=2)

        trace.extra["expanded_entities"] = list(expanded)[:20]

        # 3. Find source documents linked to expanded entities
        graph_sources = self.kg.subgraph_sources(expanded)
        trace.extra["graph_sources"] = list(graph_sources)

        # 4. Retrieve — hybrid search, then boost graph-linked chunks
        candidates = self.retriever.hybrid_search(
            query=query,
            top_k=top_k,
            domain_filter=domain_filter,
        )

        if graph_sources:
            # Boost score of chunks from graph-relevant sources
            for chunk in candidates:
                if chunk.source in graph_sources:
                    chunk.dense_score = min(1.0, chunk.dense_score * 1.25)

            # If graph found strong sources, also do targeted dense search
            # on the expanded entity text to surface additional chunks
            if expanded:
                entity_query = " ".join(list(expanded)[:10])
                extra_candidates = self.retriever.dense_search(
                    query=entity_query,
                    top_k=top_k // 2,
                    domain_filter=domain_filter,
                )
                # Merge without duplicates
                existing_ids = {c.chunk_id for c in candidates}
                for c in extra_candidates:
                    if c.chunk_id not in existing_ids:
                        candidates.append(c)
                        existing_ids.add(c.chunk_id)

        trace.dense_scores = [round(c.dense_score, 4) for c in candidates]

        # 5. Rerank on original query
        chunks = self.reranker.rerank(query, candidates)
        trace.reranker_scores = [round(c.rerank_score, 4) for c in chunks if c.rerank_score]

        # 6. Build context — inject graph relations for relevant entities
        context = self._build_graph_context(query_entities, chunks)
        confidence = self._score_confidence(query, chunks)
        trace.confidence = confidence

        # 7. Generate
        prompt = self.generator.build_rag_prompt(query, context)
        answer, _ = self.generator.generate(prompt)
        trace.tokens_estimated = self.generator.estimate_tokens(prompt)

        return RAGResponse(
            query=query,
            answer=answer,
            sources=chunks,
            strategy="graph_rag",
            trace=trace,
            latency_ms=0.0,
            confidence=confidence,
        )

    def _build_graph_context(self, entities: list, chunks: list) -> list[str]:
        """Prepend a short graph relation summary before the chunk texts."""
        context_parts = []

        # Graph relationship summary (if entities found)
        if entities:
            relations_text = []
            for entity in entities[:3]:
                rels = self.kg.get_relations(entity)[:4]
                for r in rels:
                    rel_str = ", ".join(r["relations"])
                    relations_text.append(f"{r['source']} --[{rel_str}]--> {r['target']}")
            if relations_text:
                graph_summary = "Knowledge graph context:\n" + "\n".join(relations_text)
                context_parts.append(graph_summary)

        context_parts.extend(c.text for c in chunks)
        return context_parts
