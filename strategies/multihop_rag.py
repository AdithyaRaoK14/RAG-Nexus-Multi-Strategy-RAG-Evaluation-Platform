from __future__ import annotations
import logging
from typing import Optional, List, TypedDict, Literal

import numpy as np
from langgraph.graph import StateGraph, END

from strategies.base import BaseRAGStrategy
from core.schema import RAGResponse, RAGTrace, RetrievedChunk

logger = logging.getLogger(__name__)

MAX_HOPS = 3


class HopState(TypedDict):
    original_query: str
    current_query: str
    domain_filter: Optional[str]
    all_chunks: List[dict]         # serialized RetrievedChunk dicts (LangGraph needs serializable)
    hop_answers: List[str]
    hop_count: int
    done: bool


class MultihopRAG(BaseRAGStrategy):
    """
    Multi-hop retrieval using LangGraph.

    Each hop:
      1. Retrieves on the current sub-query
      2. Generates a partial answer + new sub-question
      3. Checks if we've learned enough (cosine similarity gate)

    The final answer synthesises all accumulated chunks.
    """

    def __init__(self, config, retriever, reranker, generator):
        super().__init__(config, retriever, reranker, generator)
        self.max_hops: int = config.get("multihop", {}).get("max_hops", MAX_HOPS)
        self.sim_threshold: float = config.get("multihop", {}).get(
            "min_new_info_threshold", 0.3
        )
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # BaseRAGStrategy interface
    # ------------------------------------------------------------------

    def retrieve_and_generate(
        self,
        query: str,
        trace: RAGTrace,
        domain_filter: Optional[str],
        top_k: int,
    ) -> RAGResponse:

        initial_state: HopState = {
            "original_query": query,
            "current_query": query,
            "domain_filter": domain_filter,
            "all_chunks": [],
            "hop_answers": [],
            "hop_count": 0,
            "done": False,
        }

        final_state = self._graph.invoke(initial_state)

        # Deserialise chunks
        chunks_raw = final_state["all_chunks"]
        chunks = [self._dict_to_chunk(c) for c in chunks_raw]

        # Rerank all accumulated chunks against original query
        if chunks:
            chunks = self.reranker.rerank(query, chunks, top_k=self.cfg_ret["rerank_top_k"])

        trace.extra["hops"] = final_state["hop_count"]
        trace.extra["hop_answers"] = final_state["hop_answers"]
        trace.reranker_scores = [round(c.rerank_score, 4) for c in chunks if c.rerank_score]
        confidence = self._score_confidence(query, chunks)
        trace.confidence = confidence

        # Final synthesis
        context = self._format_context(chunks)
        synthesis_prompt = self._build_synthesis_prompt(
            original_query=query,
            hop_answers=final_state["hop_answers"],
            context=context,
        )
        answer, _ = self.generator.generate(synthesis_prompt)
        trace.tokens_estimated = self.generator.estimate_tokens(synthesis_prompt)

        return RAGResponse(
            query=query,
            answer=answer,
            sources=chunks,
            strategy="multihop_rag",
            trace=trace,
            latency_ms=0.0,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------

    def _retrieve_node(self, state: HopState) -> dict:
        """Retrieve chunks for the current sub-query."""
        candidates = self.retriever.hybrid_search(
            query=state["current_query"],
            top_k=self.cfg_ret["top_k"],
            domain_filter=state["domain_filter"],
        )
        # Only keep chunks not already in accumulated set
        existing_ids = {c["chunk_id"] for c in state["all_chunks"]}
        new_chunks = [
            self._chunk_to_dict(c) for c in candidates
            if c.chunk_id not in existing_ids
        ]
        return {"all_chunks": state["all_chunks"] + new_chunks}

    def _partial_answer_node(self, state: HopState) -> dict:
        """Generate a partial answer and the next sub-question."""
        recent_chunks = state["all_chunks"][-self.cfg_ret["top_k"]:]
        context = "\n\n".join(c["text"] for c in recent_chunks)

        prompt = (
            f"Original question: {state['original_query']}\n"
            f"Current sub-question: {state['current_query']}\n\n"
            f"Context:\n{context[:2000]}\n\n"
            f"1. Write a brief partial answer to the sub-question (2-3 sentences).\n"
            f"2. If the original question is not fully answered, write the NEXT sub-question "
            f"   that would help. Otherwise write 'DONE'.\n\n"
            f"Respond in this exact format:\n"
            f"PARTIAL: <answer>\n"
            f"NEXT: <next sub-question or DONE>"
        )
        raw, _ = self.generator.generate(prompt)
        partial, next_q = self._parse_hop_response(raw)

        new_answers = state["hop_answers"] + [partial]
        done = (next_q.upper() == "DONE") or (state["hop_count"] + 1 >= self.max_hops)

        return {
            "hop_answers": new_answers,
            "current_query": next_q if not done else state["current_query"],
            "hop_count": state["hop_count"] + 1,
            "done": done,
        }

    def _should_continue(self, state: HopState) -> Literal["retrieve", "finish"]:
        if state["done"] or state["hop_count"] >= self.max_hops:
            return "finish"
        # Check if new query is semantically different enough to be worth another hop
        if len(state["hop_answers"]) >= 2:
            sim = self._query_similarity(
                state["original_query"], state["current_query"]
            )
            if sim > (1.0 - self.sim_threshold):
                return "finish"   # sub-query is too similar, we've stopped learning
        return "retrieve"

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> any:
        builder = StateGraph(HopState)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("partial_answer", self._partial_answer_node)

        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "partial_answer")
        builder.add_conditional_edges(
            "partial_answer",
            self._should_continue,
            {"retrieve": "retrieve", "finish": END},
        )
        return builder.compile()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_synthesis_prompt(
        self, original_query: str, hop_answers: List[str], context: List[str]
    ) -> str:
        hop_summary = "\n".join(f"- {a}" for a in hop_answers)
        ctx = "\n\n---\n\n".join(context[:5])
        return (
            f"Answer the following question comprehensively using the "
            f"intermediate findings and retrieved context below.\n\n"
            f"Question: {original_query}\n\n"
            f"Intermediate findings:\n{hop_summary}\n\n"
            f"Retrieved context:\n{ctx[:3000]}\n\n"
            f"Final comprehensive answer:"
        )

    def _query_similarity(self, q1: str, q2: str) -> float:
        e1 = self.retriever.embedder.embed_query(q1)
        e2 = self.retriever.embedder.embed_query(q2)
        return float(np.dot(e1, e2))   # embeddings are normalised

    @staticmethod
    def _parse_hop_response(raw: str):
        partial, next_q = "", "DONE"
        for line in raw.splitlines():
            if line.startswith("PARTIAL:"):
                partial = line[len("PARTIAL:"):].strip()
            elif line.startswith("NEXT:"):
                next_q = line[len("NEXT:"):].strip()
        return partial or raw[:200], next_q

    @staticmethod
    def _chunk_to_dict(c: RetrievedChunk) -> dict:
        return {
            "chunk_id": c.chunk_id, "text": c.text, "source": c.source,
            "domain": c.domain, "page": c.page,
            "dense_score": c.dense_score, "sparse_score": c.sparse_score,
        }

    @staticmethod
    def _dict_to_chunk(d: dict) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=d["chunk_id"], text=d["text"], source=d["source"],
            domain=d["domain"], page=d["page"],
            dense_score=d["dense_score"], sparse_score=d["sparse_score"],
        )
