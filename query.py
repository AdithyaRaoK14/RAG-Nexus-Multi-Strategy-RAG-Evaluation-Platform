#!/usr/bin/env python3
"""
RAG-Nexus query & benchmark CLI

Usage:
  python query.py "What are SCC risk factors?"
  python query.py "What are SCC risk factors?" --strategy graph_rag
  python query.py "What are SCC risk factors?" --compare
  python query.py "What are SCC risk factors?" --model-compare
  python query.py --benchmark evaluation/benchmarks/medical.yaml
  python query.py --benchmark-all
"""
from observability.tracer import Tracer
from evaluation.benchmark import BenchmarkRunner
from evaluation.metrics import EvaluationMetrics
from knowledge_graph.graph_store import KnowledgeGraphStore
from strategies.agentic_rag import AgenticRAG
from strategies.healing_pipeline import HealingPipeline
from strategies.multihop_rag import MultihopRAG
from strategies.adaptive_rag import AdaptiveRAG
from strategies.graph_rag import GraphRAG
from strategies.advanced_rag import AdvancedRAG
from strategies.hybrid_rag import HybridRAG
from strategies.naive_rag import NaiveRAG
from core.generator import Generator
from core.reranker import Reranker
import logging
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="RAG-Nexus query interface")
console = Console()

ALL_STRATEGIES = [
    "naive_rag", "hybrid_rag", "advanced_rag",
    "graph_rag", "adaptive_rag", "multihop_rag", "agentic_rag", "healing_pipeline",
]


def _bootstrap(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )

    from core.embedder import Embedder
    from core.retriever import Retriever

    embedder = Embedder(config)
    retriever = Retriever(config, embedder)
    reranker = Reranker(config)
    generator = Generator(config)

    kg_store = KnowledgeGraphStore(config)
    tracer = Tracer(config)

    base = {
        "config": config,
        "retriever": retriever,
        "reranker": reranker,
        "generator": generator,
    }

    strategies = {
        "naive_rag": NaiveRAG(**base),
        "hybrid_rag": HybridRAG(**base),
        "advanced_rag": AdvancedRAG(**base),
        "graph_rag": GraphRAG(**base, kg_store=kg_store),
        "multihop_rag": MultihopRAG(**base),
    }

    strategies["agentic_rag"] = AgenticRAG(
        **base,
        kg_store=kg_store,
    )

    strategies["adaptive_rag"] = AdaptiveRAG(
        **base,
        strategy_registry=strategies,
    )

    strategies["healing_pipeline"] = HealingPipeline(
        **base,
        strategy_registry=strategies,
    )

    metrics = EvaluationMetrics(generator)

    runner = BenchmarkRunner(
        config,
        metrics,
    )

    for s in strategies.values():
        runner.register(s)

    return {
        "config": config,
        "strategies": strategies,
        "runner": runner,
        "tracer": tracer,
        "kg_store": kg_store,
        "generator": generator,
        "retriever": retriever,
        "reranker": reranker,
    }


@app.command()
def main(
    query: str = typer.Argument(None),
    strategy: str = typer.Option("adaptive_rag", "--strategy", "-s"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    compare: bool = typer.Option(False, "--compare", "-c"),
    model_compare: bool = typer.Option(False, "--model-compare", "-m"),
    benchmark: Optional[Path] = typer.Option(None, "--benchmark", "-b"),
    benchmark_all: bool = typer.Option(False, "--benchmark-all"),
    show_trace: bool = typer.Option(False, "--trace", "-t"),
    config_path: str = typer.Option("config.yaml", "--config"),
):
    state = _bootstrap(config_path)
    strategies = state["strategies"]
    runner = state["runner"]
    tracer = state["tracer"]
    config = state["config"]
    retriever = state["retriever"]
    reranker = state["reranker"]

    if benchmark_all:

        results = runner.run_all_benchmarks()

        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        all_results = {}

        for domain_name, domain_results in results.items():

            console.rule(f"[bold]{domain_name}[/bold]")

            runner.print_table(domain_results)

            out = output_dir / f"{domain_name}.csv"

            runner.save_csv(
                domain_results,
                out,
            )

            console.print(
                f"[green]Saved {out}[/green]"
            )

            for strat_name, strat_results in domain_results.items():

                if strat_name not in all_results:
                    all_results[strat_name] = []

                all_results[strat_name].extend(
                    strat_results
                )

        runner.save_csv(
            all_results,
            output_dir / "all_benchmarks.csv",
        )

        console.print(
            "[green]Saved results/all_benchmarks.csv[/green]"
        )

        return

    if benchmark:
        results = runner.run(benchmark, domain_filter=domain)
        runner.print_table(results)
        out = Path("results") / f"{benchmark.stem}_results.csv"
        runner.save_csv(results, out)
        console.print(f"\n[green]Saved to {out}[/green]")
        return

    if not query:
        console.print("[red]Provide a query or --benchmark[/red]")
        raise typer.Exit(1)

    if model_compare:
        _run_model_compare(
            query,
            strategy,
            domain,
            config,
            retriever,
            reranker,
        )
        return

    if compare:
        _run_strategy_compare(query, domain, strategies, tracer)
        return

    strat = strategies.get(strategy)
    if not strat:
        console.print(f"[red]Unknown strategy '{strategy}'. Choose: {ALL_STRATEGIES}[/red]")
        raise typer.Exit(1)

    response = strat.run(query=query, domain_filter=domain)
    tracer.log(response)

    console.print(Panel(response.answer, title=f"[cyan]{response.strategy}[/cyan]"))
    console.print(
        f"[dim]Confidence: {response.confidence:.3f} | "
        f"Latency: {response.latency_ms:.0f}ms[/dim]"
    )
    console.print(f"[dim]Sources: {', '.join(c.source for c in response.sources[:5])}[/dim]")
    if show_trace:
        import json
        console.print(json.dumps(response.trace.to_dict(), indent=2))


def _run_strategy_compare(query, domain, strategies, tracer):
    core = ["naive_rag", "hybrid_rag", "advanced_rag"]
    console.print(f"\n[bold]Query:[/bold] {query}\n")
    table = Table("Strategy", "Confidence", "Latency", "Sources", title="Strategy Comparison")
    for name in core:
        strat = strategies.get(name)
        if not strat:
            continue
        resp = strat.run(query=query, domain_filter=domain)
        tracer.log(resp)
        table.add_row(name, f"{resp.confidence:.3f}", f"{resp.latency_ms:.0f}ms",
                      ", ".join(c.source for c in resp.sources[:2]))
        console.print(Panel(resp.answer[:300] + "...", title=f"[cyan]{name}[/cyan]"))
    console.print(table)


def _run_model_compare(
    query,
    strategy,
    domain,
    config,
    retriever,
    reranker,
):
    from core.generator import Generator
    import copy

    models = config["models"].get(
        "comparison_models",
        [
            "qwen2.5:7b",
            "phi3:mini",
            "llama3.2:3b",
        ],
    )

    strategy_map = {
        "naive_rag": NaiveRAG,
        "hybrid_rag": HybridRAG,
        "advanced_rag": AdvancedRAG,
    }

    strategy_cls = strategy_map.get(strategy, HybridRAG)

    console.print(
        f"\n[bold]Model comparison using {strategy_cls.__name__}[/bold]\n"
    )

    table = Table(
        "Model",
        "Latency (ms)",
        "Confidence",
        "Answer preview",
        title="Model Comparison",
    )

    for model_name in models:

        mc = copy.deepcopy(config)
        mc["models"]["llm"]["default"] = model_name

        gen = Generator(mc)

        base = {
            "config": mc,
            "retriever": retriever,
            "reranker": reranker,
            "generator": gen,
        }

        # currently compare using HybridRAG



        resp = strategy_cls(**base).run(
            query=query,
            domain_filter=domain,
        )

        table.add_row(
            model_name,
            f"{resp.latency_ms:.0f}",
            f"{resp.confidence:.3f}",
            resp.answer[:80] + "...",
        )

    console.print(table)


if __name__ == "__main__":
    app()
