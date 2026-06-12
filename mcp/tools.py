from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


def make_tools(app_context: dict) -> list:
    """
    Return MCP tool definitions.
    app_context must contain: strategies, runner, tracer, kg_store, config
    """
    return [
        {
            "name": "rag_query",
            "description": (
                "Query the RAG-Nexus knowledge base. Supports all retrieval strategies: "
                "naive_rag, hybrid_rag, advanced_rag, graph_rag, adaptive_rag, "
                "multihop_rag, healing_pipeline."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question to answer"},
                    "strategy": {
                        "type": "string",
                        "enum": ["naive_rag", "hybrid_rag", "advanced_rag",
                                 "graph_rag", "adaptive_rag", "multihop_rag",
                                 "healing_pipeline"],
                        "default": "adaptive_rag",
                    },
                    "domain": {"type": "string", "description": "Filter by domain (optional)"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "compare_strategies",
            "description": "Run the same query through multiple strategies and compare results.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "strategies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["naive_rag", "hybrid_rag", "advanced_rag"],
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "run_benchmark",
            "description": "Run a benchmark YAML file against registered strategies.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "benchmark_file": {
                        "type": "string",
                        "description": "Path to benchmark YAML (e.g. evaluation/benchmarks/medical.yaml)",
                    },
                    "strategies": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["benchmark_file"],
            },
        },
        {
            "name": "evaluation_report",
            "description": "Return the latest benchmark results as a formatted table.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Filter by domain (optional)"},
                },
            },
        },
        {
            "name": "graph_query",
            "description": "Query the knowledge graph directly for entity relationships.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Entity to look up"},
                    "depth": {"type": "integer", "default": 2},
                },
                "required": ["entity"],
            },
        },
        {
            "name": "strategy_explain",
            "description": "Explain how a specific RAG strategy works and when to use it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string"},
                },
                "required": ["strategy"],
            },
        },
    ]


def execute_tool(name: str, arguments: dict, app_context: dict) -> Any:
    """Dispatch a tool call to the appropriate handler."""
    strategies = app_context["strategies"]
    runner = app_context.get("runner")
    tracer = app_context.get("tracer")
    kg_store = app_context.get("kg_store")

    if name == "rag_query":
        strat_name = arguments.get("strategy", "adaptive_rag")
        strat = strategies.get(strat_name) or strategies.get("hybrid_rag")
        response = strat.run(
            query=arguments["query"],
            domain_filter=arguments.get("domain"),
        )
        if tracer:
            tracer.log(response)
        return {
            "answer": response.answer,
            "strategy": response.strategy,
            "confidence": round(response.confidence, 3),
            "latency_ms": round(response.latency_ms, 1),
            "sources": [
                {"source": c.source, "score": round(c.final_score, 3)}
                for c in response.sources[:5]
            ],
        }

    elif name == "compare_strategies":
        results = {}
        for sname in arguments.get("strategies", ["naive_rag", "hybrid_rag", "advanced_rag"]):
            strat = strategies.get(sname)
            if not strat:
                continue
            resp = strat.run(query=arguments["query"])
            results[sname] = {
                "answer": resp.answer[:300],
                "confidence": round(resp.confidence, 3),
                "latency_ms": round(resp.latency_ms, 1),
            }
        return results

    elif name == "run_benchmark":
        if not runner:
            return {"error": "Benchmark runner not initialised"}
        results = runner.run(arguments["benchmark_file"])
        df = runner.aggregate_table(results)
        return df.to_dict(orient="records")

    elif name == "evaluation_report":
        if not tracer:
            return {"error": "Tracer not initialised"}
        return tracer.strategy_latency_summary()

    elif name == "graph_query":
        if not kg_store:
            return {"error": "Knowledge graph not built yet"}
        entity = arguments["entity"].lower()
        neighbours = list(kg_store.get_neighbors(entity, depth=arguments.get("depth", 2)))
        relations = kg_store.get_relations(entity)
        return {
            "entity": entity,
            "neighbours": neighbours[:30],
            "relations": relations[:20],
            "graph_stats": kg_store.stats(),
        }

    elif name == "strategy_explain":
        explanations = {
            "naive_rag": "Dense vector search → rerank → generate. Baseline — no query transformation.",
            "hybrid_rag": "Dense (Qdrant) + sparse (BM25) merged with RRF → rerank → generate. Best all-round.",
            "advanced_rag": "Query rewriting + HyDE + hybrid retrieval + contextual compression → generate.",
            "graph_rag": "Entity expansion via knowledge graph → boosted hybrid retrieval → generate. Best for relationship queries.",
            "adaptive_rag": "Classifies query type → routes to optimal strategy automatically.",
            "multihop_rag": "Iterative retrieval: each hop generates a new sub-query until the question is fully answered.",
            "healing_pipeline": "Tries Hybrid → Advanced → GraphRAG in sequence until confidence threshold is met.",
        }
        strat = arguments.get("strategy", "")
        return {
            "strategy": strat,
            "explanation": explanations.get(strat, "Unknown strategy"),
        }

    else:
        return {"error": f"Unknown tool: {name}"}
