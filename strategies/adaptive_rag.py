from __future__ import annotations
import logging
from typing import Optional

from strategies.base import BaseRAGStrategy
from core.schema import RAGResponse, RAGTrace

logger = logging.getLogger(__name__)

CLASSIFY_PROMPT = """\
Classify the following question into exactly one category.
Return ONLY the category name, nothing else.

Categories:
- simple             (single fact, definition, "what is X")
- entity_relationship (relationship between concepts, "how does X relate to Y")
- procedural         (step-by-step, how-to, "how to implement X")
- multi_hop          (requires chaining multiple facts, "how does A lead to B through C")
- analytical         (compare, evaluate, contrast, "compare X and Y")

Question: {query}

Category:"""


class AdaptiveRAG(BaseRAGStrategy):
    """
    Query classifier → strategy router.

    Classifies the query with a fast LLM call, then delegates to the
    most appropriate strategy. Falls back to hybrid_rag on parse failure.

    Routing table (from config.adaptive_router):
      simple             → HybridRAG
      entity_relationship → GraphRAG
      procedural         → AdvancedRAG
      multi_hop          → MultihopRAG
      analytical         → AdvancedRAG
    """

    def __init__(self, config, retriever, reranker, generator, strategy_registry: dict):
        super().__init__(config, retriever, reranker, generator)
        self.registry = strategy_registry      # name → BaseRAGStrategy instance
        self.routing_table: dict = config.get("adaptive_router", {
            "simple": "hybrid_rag",
            "entity_relationship": "graph_rag",
            "procedural": "advanced_rag",
            "multi_hop": "multihop_rag",
            "analytical": "advanced_rag",
        })

    def retrieve_and_generate(
        self,
        query: str,
        trace: RAGTrace,
        domain_filter: Optional[str],
        top_k: int,
    ) -> RAGResponse:

        # 1. Classify query
        query_type = self._classify(query)
        trace.query_type = query_type

        # 2. Route to strategy
        target_name = self.routing_table.get(query_type, "hybrid_rag")
        target = self.registry.get(target_name) or self.registry.get("hybrid_rag")

        trace.router_scores = {qt: (1.0 if qt == query_type else 0.0)
                               for qt in self.routing_table}
        trace.extra["routed_to"] = target_name
        logger.info(f"AdaptiveRAG: '{query_type}' → {target_name}")

        # 3. Delegate — run the selected strategy's core logic directly
        #    (avoids double-timing; base.run() wraps the whole thing)
        inner_trace = RAGTrace(
            query=query,
            strategy=target_name,
            query_type=query_type,
        )
        response = target.retrieve_and_generate(
            query=query,
            trace=inner_trace,
            domain_filter=domain_filter,
            top_k=top_k,
        )

        # Merge inner trace into adaptive trace
        trace.reranker_scores = inner_trace.reranker_scores
        trace.dense_scores = inner_trace.dense_scores
        trace.sparse_scores = inner_trace.sparse_scores
        trace.tokens_estimated = inner_trace.tokens_estimated
        trace.confidence = inner_trace.confidence
        trace.extra.update(inner_trace.extra)

        response.strategy = "adaptive_rag"
        response.trace = trace
        return response

    # ------------------------------------------------------------------

    def _classify(self, query: str) -> str:
        prompt = CLASSIFY_PROMPT.format(query=query)
        try:
            raw, _ = self.generator.generate(prompt)
            return self._parse_type(raw.strip().lower())
        except Exception as e:
            logger.warning(f"Classification failed: {e}. Defaulting to 'simple'")
            return "simple"

    @staticmethod
    def _parse_type(raw: str) -> str:
        valid = {"simple", "entity_relationship", "procedural", "multi_hop", "analytical"}
        # Direct match
        for t in valid:
            if t in raw:
                return t
        return "simple"
