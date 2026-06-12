#!/usr/bin/env python3
"""
RAG-Nexus MCP Server (stdio transport, JSON-RPC 2.0)

Exposes RAG-Nexus's own retrieval tools as MCP tools.
This is a standalone server — connect to any MCP-compatible client
when you want to, or run the tools directly via query.py CLI.

Start the server:
  python -m mcp.server

Then configure any MCP client to point to this process.
"""
from __future__ import annotations
import json
import logging
import sys
import yaml

from mcp.tools import make_tools, execute_tool

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _load_app(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

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
    from evaluation.metrics import EvaluationMetrics
    from evaluation.benchmark import BenchmarkRunner
    from observability.tracer import Tracer

    embedder = Embedder(config)
    retriever = Retriever(config, embedder)
    reranker = Reranker(config)
    generator = Generator(config)
    kg_store = KnowledgeGraphStore(config)
    tracer = Tracer(config)

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

    return dict(strategies=strategies, runner=runner, tracer=tracer,
                kg_store=kg_store, config=config)


def _send(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def _error(id_: any, code: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}})


def serve(config_path: str = "config.yaml") -> None:
    logger.info("RAG-Nexus MCP server starting...")
    app_context = _load_app(config_path)
    tools = make_tools(app_context)
    logger.info(f"MCP server ready — {len(tools)} tools")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _error(None, -32700, "Parse error")
            continue

        msg_id = msg.get("id")
        method  = msg.get("method", "")

        try:
            if method == "initialize":
                _send({
                    "jsonrpc": "2.0", "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "rag-nexus", "version": "1.0.0"},
                    },
                })
            elif method == "notifications/initialized":
                pass
            elif method == "tools/list":
                _send({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": tools}})
            elif method == "tools/call":
                params = msg.get("params", {})
                result = execute_tool(params.get("name"), params.get("arguments", {}), app_context)
                _send({
                    "jsonrpc": "2.0", "id": msg_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
                })
            elif method == "ping":
                _send({"jsonrpc": "2.0", "id": msg_id, "result": {}})
            else:
                _error(msg_id, -32601, f"Method not found: {method}")
        except Exception as e:
            logger.error(f"Error handling {method}: {e}", exc_info=True)
            _error(msg_id, -32603, str(e))


if __name__ == "__main__":
    serve()
