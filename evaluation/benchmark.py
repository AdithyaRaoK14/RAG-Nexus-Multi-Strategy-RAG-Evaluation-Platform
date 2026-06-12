from __future__ import annotations
import json
import logging
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

from core.schema import BenchmarkQuery, BenchmarkResult
from evaluation.metrics import EvaluationMetrics
from strategies.base import BaseRAGStrategy

logger = logging.getLogger(__name__)
console = Console()


class BenchmarkRunner:
    """
    Loads benchmark YAML files, runs registered strategies,
    and produces a comparison table.

    Benchmark YAML format:
      - query: "What are the major risk factors for SCC?"
        expected_docs:
          - scc_paper_02.pdf
          - scc_paper_07.pdf
        query_type: entity_relationship
        domain: medical
        expected_answer: "..."   # optional

    Usage:
        runner = BenchmarkRunner(config, metrics)
        runner.register(naive_rag)
        runner.register(hybrid_rag)
        runner.register(advanced_rag)
        results = runner.run("evaluation/benchmarks/medical.yaml")
        runner.print_table(results)
        runner.save_csv(results, "results/medical_benchmark.csv")
    """

    def __init__(self, config: dict, metrics: EvaluationMetrics):
        self.config = config
        self.metrics = metrics
        self.strategies: Dict[str, BaseRAGStrategy] = {}

    def register(self, strategy: BaseRAGStrategy) -> None:
        self.strategies[strategy.strategy_name] = strategy
        logger.info(f"Registered strategy: {strategy.strategy_name}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        benchmark_path: str | Path,
        domain_filter: Optional[str] = None,
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Run all registered strategies against all queries in the YAML.
        Returns dict of strategy_name → List[BenchmarkResult].
        """
        queries, metadata = self._load_yaml(Path(benchmark_path))
        if not queries:
            raise ValueError(f"No queries found in {benchmark_path}")

        if not self.strategies:
            raise RuntimeError(
                "No strategies registered. Call runner.register() first.")

        console.print(f"\n[bold]Benchmark:[/bold] {Path(benchmark_path).name}")
        if metadata:
            console.print(
                f"[dim]type={metadata.get('benchmark_type', '?')}  "
                f"version={metadata.get('version', '?')}[/dim]"
            )
        console.print(
            f"Queries: {len(queries)} | Strategies: {list(self.strategies.keys())}\n")

        all_results: Dict[str, List[BenchmarkResult]] = {
            s: [] for s in self.strategies}

        for i, bq in enumerate(queries):
            console.print(f"  [{i+1}/{len(queries)}] {bq.query[:60]}...")
            df = domain_filter or bq.domain

            for strat_name, strategy in self.strategies.items():
                try:
                    response = strategy.run(query=bq.query, domain_filter=df)
                    result = self.metrics.evaluate(response, bq)
                    all_results[strat_name].append(result)
                    console.print(
                        f"    {strat_name:20s} | "
                        f"Hit@5={result.hit_at_5:.2f} "
                        f"MRR={result.mrr:.3f} "
                        f"Recall@5={result.recall_at_5:.2f} "
                        f"Faith={result.faithfulness:.2f} "
                        f"{result.latency_ms:.0f}ms"
                    )
                except Exception as e:
                    logger.error(
                        f"Strategy {strat_name} failed on query: {bq.query[:40]} — {e}")

        return all_results

    def run_all_benchmarks(
        self,
        benchmarks_dir: str | Path = "evaluation/benchmarks",
    ) -> Dict[str, Dict[str, List[BenchmarkResult]]]:
        """Run every YAML file in benchmarks_dir."""
        results = {}
        for yaml_path in sorted(Path(benchmarks_dir).glob("*.yaml")):
            domain = yaml_path.stem
            results[domain] = self.run(yaml_path)
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def aggregate_table(
        self,
        results: Dict[str, List[BenchmarkResult]],
    ) -> pd.DataFrame:
        """Build a summary DataFrame: rows = strategies, cols = avg metrics."""
        rows = []
        for strat_name, result_list in results.items():
            if not result_list:
                continue
            agg = self.metrics.aggregate(result_list)
            rows.append({
                "strategy": strat_name,
                "Hit@1": agg.get("hit_at_1", 0),
                "Hit@3": agg.get("hit_at_3", 0),
                "Hit@5": agg.get("hit_at_5", 0),
                "MRR": agg.get("mrr", 0),
                "Recall@5": agg.get("recall_at_5", 0),
                "Recall@10": agg.get("recall_at_10", 0),
                "Faithfulness": agg.get("faithfulness", 0),
                "Ctx Precision": agg.get("context_precision", 0),
                "Latency (ms)": agg.get("latency_ms", 0),
                "N": agg.get("n", 0),
            })
        return pd.DataFrame(rows).sort_values("MRR", ascending=False)

    def print_table(self, results: Dict[str, List[BenchmarkResult]]) -> None:
        df = self.aggregate_table(results)
        table = Table(title="RAG-Nexus Benchmark Results", show_lines=True)

        for col in df.columns:
            table.add_column(col, justify="right" if col !=
                             "strategy" else "left")

        for _, row in df.iterrows():
            table.add_row(*[
                str(row[c]) if c == "strategy" else
                f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c])
                for c in df.columns
            ])

        console.print(table)

    def save_csv(
        self,
        results: Dict[str, List[BenchmarkResult]],
        output_path: str | Path,
        benchmark_path: str | Path | None = None,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.aggregate_table(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

        # Companion config — answers "which run produced this CSV?"
        cfg = self.config.get("models", {})
        config_path = output_path.with_name(output_path.stem + "_config.json")
        run_config = {
            "date": datetime.now(timezone.utc).isoformat(),
            "benchmark": str(benchmark_path) if benchmark_path else None,
            "strategies": list(results.keys()),
            "embedding_model": cfg.get("embedding", {}).get("name", "unknown"),
            "reranker": cfg.get("reranker", {}).get("name", "unknown"),
            "llm": cfg.get("llm", {}).get("default", "unknown"),
            "n_queries": sum(len(v) for v in results.values()),
        }
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2)
        logger.info(f"Run config saved to {config_path}")

    def print_leaderboard(
        self,
        results: Dict[str, List[BenchmarkResult]],
    ) -> None:
        """
        Composite score with proper unit normalisation:
          score = 0.4*MRR + 0.3*faithfulness + 0.2*recall@5 - 0.1*latency_norm
        latency_norm = min-max scaled to [0,1] across strategies.
        """
        df = self.aggregate_table(results)
        if df.empty:
            return

        lat_min = df["Latency (ms)"].min()
        lat_max = df["Latency (ms)"].max()
        lat_range = lat_max - lat_min or 1.0   # avoid div-by-zero when all equal

        df["latency_norm"] = (df["Latency (ms)"] - lat_min) / lat_range

        recall_col = "Recall@5" if "Recall@5" in df.columns else None
        recall_vals = df[recall_col] if recall_col else 0.0

        df["score"] = (
            0.4 * df["MRR"]
            + 0.3 * df["Faithfulness"]
            + 0.2 * recall_vals
            - 0.1 * df["latency_norm"]
        ).clip(0, 1)

        df = df.sort_values("score", ascending=False).reset_index(drop=True)

        table = Table(title="🏆 RAG-Nexus Leaderboard", show_lines=True)
        table.add_column("Rank", justify="center")
        table.add_column("Strategy", justify="left")
        table.add_column("MRR", justify="right")
        table.add_column("Faithfulness", justify="right")
        table.add_column("Recall@5", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Score", justify="right")

        medals = {0: "🥇", 1: "🥈", 2: "🥉"}
        for i, row in df.iterrows():
            rank = medals.get(i, str(i + 1))
            table.add_row(
                rank,
                str(row["strategy"]),
                f"{row['MRR']:.3f}",
                f"{row['Faithfulness']:.3f}",
                f"{row[recall_col]:.3f}" if recall_col else "—",
                f"{row['Latency (ms)']:.0f}",
                f"{row['score']:.3f}",
            )

        console.print(table)

    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> tuple[List[BenchmarkQuery], dict]:
        with open(path) as f:
            raw = yaml.safe_load(f)

        metadata = {}
        queries = []
        for item in raw:
            if isinstance(item, dict) and "metadata" in item and len(item) == 1:
                metadata = item["metadata"]
                continue
            queries.append(BenchmarkQuery(
                query=item["query"],
                expected_docs=item.get("expected_docs", []),
                query_type=item.get("query_type", "simple"),
                domain=item.get("domain", ""),
                expected_answer=item.get("expected_answer"),
            ))
        return queries, metadata
