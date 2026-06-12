from __future__ import annotations
import logging
import time
from typing import Optional, List, TypedDict, Literal

from langgraph.graph import StateGraph, END

from strategies.base import BaseRAGStrategy
from core.schema import RAGResponse, RAGTrace

logger = logging.getLogger(__name__)


class HealingState(TypedDict):
    query: str
    domain_filter: Optional[str]
    top_k: int
    confidence_threshold: float
    cascade: List[str]            # ordered strategy names to try
    attempt_index: int            # which strategy we're on
    responses: List[dict]         # serialized RAGResponse summaries
    best_response_idx: int
    healed: bool                  # true if we had to fall back


class HealingPipeline(BaseRAGStrategy):
    """
    Self-healing retrieval pipeline.

    Tries strategies in order (default: hybrid → advanced → graph).
    If a strategy's confidence < threshold, moves to the next one.
    Returns the response with the highest confidence score.

    On your resume:
    "Designed a self-healing RAG pipeline that automatically cascaded
     through Hybrid, Advanced, and GraphRAG strategies when retrieval
     confidence fell below a configurable threshold."
    """

    def __init__(self, config, retriever, reranker, generator, strategy_registry: dict):
        super().__init__(config, retriever, reranker, generator)
        self.registry = strategy_registry
        self.cascade_config: List[str] = (
            config.get("healing_pipeline", {}).get("cascade", [
                "hybrid_rag", "advanced_rag", "graph_rag"
            ])
        )
        self.threshold: float = config.get("retrieval", {}).get(
            "confidence_threshold", 0.55
        )
        self._lg = self._build_graph()

    # ------------------------------------------------------------------

    def retrieve_and_generate(
        self,
        query: str,
        trace: RAGTrace,
        domain_filter: Optional[str],
        top_k: int,
    ) -> RAGResponse:


        t0 = time.perf_counter()

        initial: HealingState = {
            "query": query,
            "domain_filter": domain_filter,
            "top_k": top_k,
            "confidence_threshold": self.threshold,
            "cascade": self.cascade_config,
            "attempt_index": 0,
            "responses": [],
            "best_response_idx": 0,
            "healed": False,
        }

        final = self._lg.invoke(initial)

        # Pick the best response
        best_summary = final["responses"][final["best_response_idx"]]
        trace.healing_attempts = [r["strategy"] for r in final["responses"]]
        trace.extra["healed"] = final["healed"]
        trace.extra["all_confidences"] = [r["confidence"] for r in final["responses"]]
        trace.confidence = best_summary["confidence"]

        # Re-run the winning strategy to get full RAGResponse
        winning_name = best_summary["strategy"]
        winner = self.registry.get(winning_name) or list(self.registry.values())[0]
        inner_trace = RAGTrace(
            query=query,
            strategy=winning_name,
            healing_attempts=trace.healing_attempts,
        )
        response = winner.retrieve_and_generate(query, inner_trace, domain_filter, top_k)
        response.latency_ms = (
            time.perf_counter() - t0
        ) * 1000
        response.strategy = "healing_pipeline"
        response.trace = trace
        trace.tokens_estimated = inner_trace.tokens_estimated
        return response

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------

    def _attempt_node(self, state: HealingState) -> dict:
        """Run the current strategy, compute confidence, store summary."""
        strat_name = state["cascade"][state["attempt_index"]]
        strategy = self.registry.get(strat_name)

        if strategy is None:
            logger.warning(f"Strategy '{strat_name}' not in registry, skipping")
            confidence = 0.0
        else:
            inner_trace = RAGTrace(query=state["query"], strategy=strat_name)
            try:
                resp = strategy.retrieve_and_generate(
                    query=state["query"],
                    trace=inner_trace,
                    domain_filter=state["domain_filter"],
                    top_k=state["top_k"],
                )
                confidence = resp.confidence
                summary = {
                    "strategy": strat_name,
                    "confidence": confidence,
                    "answer_preview": resp.answer[:200],
                }
            except Exception as e:
                logger.error(f"Strategy {strat_name} failed during healing: {e}")
                confidence = 0.0
                summary = {"strategy": strat_name, "confidence": 0.0, "answer_preview": ""}

        responses = state["responses"] + [summary]

        # Track best so far
        best_idx = max(range(len(responses)), key=lambda i: responses[i]["confidence"])

        return {
            "responses": responses,
            "best_response_idx": best_idx,
            "healed": state["attempt_index"] > 0,
        }

    def _should_heal(self, state: HealingState) -> Literal["heal", "done"]:
        last_conf = state["responses"][-1]["confidence"]
        next_idx = state["attempt_index"] + 1

        if last_conf >= state["confidence_threshold"]:
            return "done"
        if next_idx >= len(state["cascade"]):
            return "done"
        return "heal"

    def _advance_node(self, state: HealingState) -> dict:
        return {"attempt_index": state["attempt_index"] + 1}

    # ------------------------------------------------------------------

    def _build_graph(self):
        builder = StateGraph(HealingState)
        builder.add_node("attempt", self._attempt_node)
        builder.add_node("advance", self._advance_node)

        builder.set_entry_point("attempt")
        builder.add_conditional_edges(
            "attempt",
            self._should_heal,
            {"heal": "advance", "done": END},
        )
        builder.add_edge("advance", "attempt")
        return builder.compile()
