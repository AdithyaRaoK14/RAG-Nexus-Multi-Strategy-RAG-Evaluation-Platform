from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class QueryType(str, Enum):
    SIMPLE = "simple"                          # single-fact lookups
    ENTITY_RELATIONSHIP = "entity_relationship"  # what relates to what
    PROCEDURAL = "procedural"                  # how-to, step-by-step
    MULTI_HOP = "multi_hop"                    # requires chained reasoning
    ANALYTICAL = "analytical"                  # compare / contrast / evaluate


class StrategyType(str, Enum):
    NAIVE = "naive_rag"
    HYBRID = "hybrid_rag"
    ADVANCED = "advanced_rag"
    GRAPH = "graph_rag"
    ADAPTIVE = "adaptive_rag"
    MULTIHOP = "multihop_rag"


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str          # filename e.g. "scc_paper_03.pdf"
    domain: str          # e.g. "medical"
    page: int
    dense_score: float
    sparse_score: float = 0.0
    rerank_score: Optional[float] = None

    @property
    def final_score(self) -> float:
        """Rerank score when available, else dense score."""
        return self.rerank_score if self.rerank_score is not None else self.dense_score

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "domain": self.domain,
            "page": self.page,
            "dense_score": round(self.dense_score, 4),
            "sparse_score": round(self.sparse_score, 4),
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score else None,
            "final_score": round(self.final_score, 4),
            "text_preview": self.text[:200] + ("..." if len(self.text) > 200 else ""),
        }


@dataclass
class RAGTrace:
    """Captures every decision the pipeline made — feeds Streamlit observability."""
    query: str
    strategy: str
    query_type: Optional[str] = None
    router_scores: Dict[str, float] = field(default_factory=dict)
    # Advanced RAG query rewriting
    rewrite_query: Optional[str] = None
    hyde_doc: Optional[str] = None                # HyDE hypothetical document
    dense_scores: List[float] = field(default_factory=list)
    sparse_scores: List[float] = field(default_factory=list)
    reranker_scores: List[float] = field(default_factory=list)
    confidence: float = 0.0
    healing_attempts: List[str] = field(
        default_factory=list)  # fallback strategies tried
    tokens_estimated: int = 0
    latency_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "query_type": self.query_type,
            "router_scores": self.router_scores,
            "rewrite_query": self.rewrite_query,
            "hyde_doc": self.hyde_doc,
            "dense_scores": self.dense_scores,
            "sparse_scores": self.sparse_scores,
            "reranker_scores": self.reranker_scores,
            "confidence": round(self.confidence, 4),
            "healing_attempts": self.healing_attempts,
            "tokens_estimated": self.tokens_estimated,
            "latency_ms": round(self.latency_ms, 2),
            **self.extra,
        }


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: List[RetrievedChunk]
    strategy: str
    trace: RAGTrace
    latency_ms: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [c.to_dict() for c in self.sources],
            "strategy": self.strategy,
            "latency_ms": round(self.latency_ms, 2),
            "confidence": round(self.confidence, 4),
            "trace": self.trace.to_dict(),
        }


@dataclass
class BenchmarkQuery:
    query: str
    expected_docs: List[str]         # filenames that should be retrieved
    query_type: str
    domain: str
    # optional gold answer for faithfulness
    expected_answer: Optional[str] = None


@dataclass
class BenchmarkResult:
    query: str
    strategy: str
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    faithfulness: float
    context_precision: float
    latency_ms: float
    retrieved_sources: List[str]
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query[:60] + "..." if len(self.query) > 60 else self.query,
            "strategy": self.strategy,
            "hit@1": round(self.hit_at_1, 3),
            "hit@3": round(self.hit_at_3, 3),
            "hit@5": round(self.hit_at_5, 3),
            "mrr": round(self.mrr, 3),
            "recall@5": round(self.recall_at_5, 3),
            "recall@10": round(self.recall_at_10, 3),
            "faithfulness": round(self.faithfulness, 3),
            "ctx_precision": round(self.context_precision, 3),
            "latency_ms": round(self.latency_ms, 1),
        }
