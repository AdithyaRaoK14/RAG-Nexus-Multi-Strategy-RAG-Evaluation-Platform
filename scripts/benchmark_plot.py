"""
scripts/benchmark_plot.py

Generates benchmark visualisations from a benchmark results CSV.

Usage:
    python scripts/benchmark_plot.py

    python scripts/benchmark_plot.py \
        --csv results/all_benchmarks.csv

    python scripts/benchmark_plot.py \
        --csv results/all_benchmarks.csv \
        --metrics mrr faithfulness recall_at_5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    "mrr",
    "hit@1",
    "hit@3",
    "hit@5",
    "recall@5",
    "recall@10",
    "faithfulness",
    "ctx precision",
    "latency (ms)",
]


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    """
    Generate a horizontal bar chart for one metric.
    """

    col_map = {c.lower(): c for c in df.columns}
    col = col_map.get(metric.lower())

    if col is None:
        print(
            f"  [skip] '{metric}' not found "
            f"in CSV columns: {list(df.columns)}"
        )
        return


    avg = (
        df.groupby("strategy")[col]
        .mean()
        .sort_values(
            ascending=(metric.lower() != "latency (ms)")
        )
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    avg.plot.barh(
        ax=ax,
        edgecolor="white",
    )

    ax.set_xlabel(col)
    ax.set_title(col.replace("_", " ").title())

    if metric != "latency (ms)":
        ax.set_xlim(0, 1)

    for bar in ax.patches:
        ax.text(
            bar.get_width() + (
                0.01 if metric != "latency (ms)"
                else max(avg) * 0.01
            ),
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()

    safe_name = (
        metric.replace("@", "_at_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


    out = output_dir / f"{safe_name}.png"

    plt.savefig(
        out,
        dpi=150,
    )

    plt.close()

    print(f"  saved → {out}")


def plot_latency_vs_mrr(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate latency vs MRR tradeoff scatter plot.
    """

    required = {
        "strategy",
        "latency (ms)",
        "mrr",
    }

    if not required.issubset(df.columns):
        print(
            "  [skip] Cannot generate latency_vs_mrr.png "
            f"(missing {required - set(df.columns)})"
        )
        return

    avg = (
        df.groupby("strategy")
        .agg(
            latency_ms=("latency (ms)", "mean"),
            mrr=("mrr", "mean"),
        )
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        avg["latency_ms"],
        avg["mrr"],
    )

    for name, row in avg.iterrows():
        plt.annotate(
            name,
            (
                row["latency_ms"],
                row["mrr"],
            ),
        )

    plt.xlabel("Latency (ms)")
    plt.ylabel("MRR")
    plt.title("Latency vs Retrieval Quality")

    plt.tight_layout()

    out = output_dir / "latency_vs_mrr.png"

    plt.savefig(
        out,
        dpi=150,
    )

    plt.close()

    print(f"  saved → {out}")


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default="results/all_benchmarks.csv",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=METRICS,
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    df.columns = [
        c.lower()
        for c in df.columns
    ]

    output_dir = csv_path.parent / "plots"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Plotting {len(args.metrics)} metrics "
        f"from {csv_path}"
    )

    for metric in args.metrics:
        plot_metric(
            df,
            metric,
            output_dir,
        )

    plot_latency_vs_mrr(
        df,
        output_dir,
    )

    print("\nDone.")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
