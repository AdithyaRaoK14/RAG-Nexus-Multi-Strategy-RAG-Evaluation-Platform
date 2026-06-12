from __future__ import annotations
import logging
import yaml
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.query import router as query_router
from api.routes.ingest import router as ingest_router
from api.routes.eval import router as eval_router

logger = logging.getLogger(__name__)
_state: dict = {}


def _bootstrap(config: dict) -> dict:
    from core.embedder import Embedder
    from core.retriever import Retriever
    from core.reranker import Reranker
    from core.generator import Generator
    from strategies.naive_rag import NaiveRAG
    from strategies.hybrid_rag import HybridRAG
    from strategies.advanced_rag import AdvancedRAG
    from strategies.graph_rag import GraphRAG
    from strategies.adaptive_rag import AdaptiveRAG
    from strategies.multihop_rag import MultihopRAG
    from strategies.agentic_rag import AgenticRAG
    from strategies.healing_pipeline import HealingPipeline
    from knowledge_graph.graph_store import KnowledgeGraphStore
    from ingestion.indexers import Indexer
    from evaluation.metrics import EvaluationMetrics
    from evaluation.benchmark import BenchmarkRunner
    from observability.tracer import Tracer

    embedder = Embedder(config)
    retriever = Retriever(config, embedder)
    reranker = Reranker(config)
    generator = Generator(config)
    kg_store = KnowledgeGraphStore(config)
    tracer = Tracer(config)
    indexer = Indexer(config, retriever)

    base = dict(config=config, retriever=retriever, reranker=reranker, generator=generator)
    strategies = {
        "naive_rag":    NaiveRAG(**base),
        "hybrid_rag":   HybridRAG(**base),
        "advanced_rag": AdvancedRAG(**base),
        "graph_rag":    GraphRAG(**base, kg_store=kg_store),
        "multihop_rag": MultihopRAG(**base),
        "agentic_rag":  AgenticRAG(**base, kg_store=kg_store),
    }
    strategies["adaptive_rag"]    = AdaptiveRAG(**base, strategy_registry=strategies)
    strategies["healing_pipeline"] = HealingPipeline(**base, strategy_registry=strategies)

    metrics = EvaluationMetrics(generator)
    runner = BenchmarkRunner(config, metrics)
    for s in strategies.values():
        runner.register(s)

    return dict(strategies=strategies, retriever=retriever, indexer=indexer,
                kg_store=kg_store, runner=runner, tracer=tracer, config=config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])
    logger.info("Bootstrapping RAG-Nexus API...")
    _state.update(_bootstrap(config))
    logger.info("API ready")
    yield
    _state.clear()


def get_state() -> dict:
    return _state


app = FastAPI(
    title="RAG-Nexus API",
    description="Multi-strategy RAG evaluation platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(query_router, prefix="/query", tags=["Query"])
app.include_router(ingest_router, prefix="/ingest", tags=["Ingest"])
app.include_router(eval_router, prefix="/eval", tags=["Evaluation"])


@app.get("/health")
def health():
    return {"status": "ok", "strategies": list(_state.get("strategies", {}).keys())}


@app.get("/stats")
def stats():
    r = _state.get("retriever")
    kg = _state.get("kg_store")
    return {
        "index": r.collection_stats() if r else {},
        "knowledge_graph": kg.stats() if kg else {},
    }
