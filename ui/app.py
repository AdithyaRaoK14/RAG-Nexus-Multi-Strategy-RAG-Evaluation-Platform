"""
RAG-Nexus Streamlit Dashboard

Run: streamlit run ui/app.py
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path


import streamlit as st
import pandas as pd
import plotly.express as px
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="RAG-Nexus",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Resource initialisation (cached)
# ------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading RAG-Nexus components...")
def load_components():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)

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
    from strategies.healing_pipeline import HealingPipeline
    from strategies.agentic_rag import AgenticRAG
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

    base = {"config": config, "retriever": retriever,
            "reranker": reranker, "generator": generator}

    strategies = {
        "naive_rag":    NaiveRAG(**base),
        "hybrid_rag":   HybridRAG(**base),
        "advanced_rag": AdvancedRAG(**base),
        "graph_rag":    GraphRAG(**base, kg_store=kg_store),
        "multihop_rag": MultihopRAG(**base),
        "agentic_rag":  AgenticRAG(**base, kg_store=kg_store),
    }
    strategies["adaptive_rag"] = AdaptiveRAG(
        **base, strategy_registry=strategies)
    strategies["healing_pipeline"] = HealingPipeline(
        **base, strategy_registry=strategies)

    metrics = EvaluationMetrics(generator)
    runner = BenchmarkRunner(config, metrics)
    for s in strategies.values():
        runner.register(s)
    logging.info(
        f"Registered strategies: {list(runner.strategies.keys())}"
    )

    return {
        "strategies": strategies, "runner": runner,
        "tracer": tracer, "kg_store": kg_store, "config": config,
    }


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

st.sidebar.title("🔍 RAG-Nexus")
st.sidebar.caption("Multi-strategy RAG evaluation platform")

tab = st.sidebar.radio(
    "Navigate",
    ["Query", "Benchmark", "Observability", "Knowledge Graph"],
    index=0,
)

components = load_components()

# ------------------------------------------------------------------
# Tab 1: Query
# ------------------------------------------------------------------

if tab == "Query":
    st.title("Query Interface")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter your question",
                              placeholder="What are the risk factors for SCC?")
    with col2:
        strategy = st.selectbox(
            "Strategy",
            list(components["strategies"].keys()),
            index=6,   # adaptive_rag
        )

    domain_options = ["(all domains)", "medical", "security", "rag_retrieval",
                      "llm", "system_design", "devops", "aiml_cloud"]
    domain_sel = st.selectbox("Domain filter", domain_options)
    domain = None if domain_sel == "(all domains)" else domain_sel

    show_trace = st.checkbox("Show trace details")

    if st.button("Run Query", type="primary") and query:
        strat = components["strategies"][strategy]
        with st.spinner(f"Running {strategy}..."):
            response = strat.run(query=query, domain_filter=domain)
            components["tracer"].log(response)

        # Answer
        st.subheader("Answer")
        st.markdown(response.answer)

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{response.confidence:.3f}")
        c2.metric("Latency", f"{response.latency_ms:.0f} ms")
        c3.metric("Strategy", response.strategy)

        # Sources
        st.subheader("Retrieved Sources")
        src_data = [
            {
                "Source": c.source,
                "Domain": c.domain,
                "Page": c.page,
                "Score": f"{c.final_score:.4f}",
                "Preview": c.text[:120] + "...",
            }
            for c in response.sources
        ]
        st.dataframe(pd.DataFrame(src_data), use_container_width=True)

        # Reranker scores chart
        if response.trace.reranker_scores:
            fig = px.bar(
                x=[f"Chunk {i+1}" for i in range(
                    len(response.trace.reranker_scores))],
                y=response.trace.reranker_scores,
                labels={"x": "Chunk", "y": "Reranker Score"},
                title="Reranker Scores",
                color=response.trace.reranker_scores,
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Trace
        if show_trace:
            with st.expander("Full Trace"):
                st.json(response.trace.to_dict())

        # Self-healing info
        if response.trace.healing_attempts and len(response.trace.healing_attempts) > 1:
            st.info(
                f"🔧 Self-healing triggered — tried: "
                f"{' → '.join(response.trace.healing_attempts)}"
            )

# ------------------------------------------------------------------
# Tab 2: Benchmark
# ------------------------------------------------------------------

elif tab == "Benchmark":
    st.title("Benchmark Dashboard")

    bench_files = sorted(Path("evaluation/benchmarks").glob("*.yaml"))
    bench_names = [f.name for f in bench_files]

    if not bench_names:
        st.warning("No benchmark YAML files found.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        selected_bench = st.selectbox(
            "Benchmark file",
            bench_names,
        )

    runner = components["runner"]

    with col2:
        strat_options = list(runner.strategies.keys())

        selected_strats = st.multiselect(
            "Strategies to benchmark",
            strat_options,
            default=strat_options[:3],
        )

    if st.button("Run Benchmark", type="primary"):

        if not selected_strats:
            st.error("Please select at least one strategy.")
            st.stop()

        original_strategies = runner.strategies.copy()
        results = None

        try:
            # keep only selected strategies
            runner.strategies = {
                name: strategy
                for name, strategy in original_strategies.items()
                if name in selected_strats
            }

            if not runner.strategies:
                st.error(
                    "Selected strategies could not be found. "
                    f"Available: {list(original_strategies.keys())}"
                )
                st.stop()

            bench_path = Path("evaluation/benchmarks") / selected_bench

            with st.spinner(
                f"Running benchmark on {len(runner.strategies)} strategies..."
            ):
                results = runner.run(bench_path)

        except Exception as e:
            st.exception(e)

        finally:
            # always restore full registry
            runner.strategies = original_strategies

        if results is not None:

            df = runner.aggregate_table(results)

            st.session_state["last_bench_df"] = df
            st.session_state["last_bench_results"] = results

            st.success("Benchmark completed successfully.")

    # --------------------------------------------------------------
    # Display previous benchmark results
    # --------------------------------------------------------------

    if "last_bench_df" in st.session_state:

        df = st.session_state["last_bench_df"]
        results = st.session_state.get("last_bench_results", {})

        st.subheader("Aggregate Results")
        st.dataframe(df, use_container_width=True)

        if not df.empty:

            col_left, col_right = st.columns(2)

            with col_left:
                fig = px.bar(df, x="strategy", y="MRR", color="strategy",
                             title="MRR by Strategy", text_auto=".3f")
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                fig2 = px.bar(df, x="strategy", y="Faithfulness", color="strategy",
                              title="Faithfulness by Strategy", text_auto=".3f")
                st.plotly_chart(fig2, use_container_width=True)

            # Recall@5 / Recall@10
            if "Recall@5" in df.columns:
                recall_df = df[["strategy", "Recall@5", "Recall@10"]].melt(
                    id_vars="strategy", var_name="k", value_name="recall"
                )
                fig_r = px.bar(recall_df, x="strategy", y="recall", color="k",
                               barmode="group", title="Recall@k by Strategy",
                               text_auto=".3f")
                st.plotly_chart(fig_r, use_container_width=True)

            fig3 = px.scatter(df, x="Latency (ms)", y="Faithfulness",
                              text="strategy", size="MRR",
                              title="Quality vs Latency")
            st.plotly_chart(fig3, use_container_width=True)

            # Leaderboard
            if results:
                st.subheader("🏆 Leaderboard")
                lat_min = df["Latency (ms)"].min()
                lat_max = df["Latency (ms)"].max()
                lat_range = lat_max - lat_min or 1.0
                lb = df.copy()

                lb["latency_norm"] = (
                    lb["Latency (ms)"] - lat_min
                ) / lat_range

                recall_vals = (
                    lb["Recall@5"]
                    if "Recall@5" in lb.columns
                    else 0.0
                )

                lb["score"] = (
                    0.4 * lb["MRR"]
                    + 0.3 * lb["Faithfulness"]
                    + 0.2 * recall_vals
                    - 0.1 * lb["latency_norm"]
                ).clip(0, 1)

                lb = lb[
                    ["strategy", "MRR", "Faithfulness",
                    "Latency (ms)", "score"]
                ].sort_values(
                    "score",
                    ascending=False,
                ).reset_index(drop=True)
                lb.index += 1
                lb.index.name = "Rank"
                st.dataframe(lb.style.format({
                    "MRR": "{:.3f}", "Faithfulness": "{:.3f}",
                    "Latency (ms)": "{:.0f}", "score": "{:.3f}",
                }), use_container_width=True)

            csv = df.drop(columns=["latency_norm", "score"],
                          errors="ignore").to_csv(index=False)
            st.download_button("Download CSV", csv,
                               "benchmark_results.csv", "text/csv")

# ------------------------------------------------------------------
# Tab 3: Observability
# ------------------------------------------------------------------

elif tab == "Observability":
    st.title("Observability")
    tracer = components["tracer"]

    subtab = st.radio("View", ["Recent Traces", "Latency Summary", "Self-Healing Cases"],
                      horizontal=True)

    if subtab == "Recent Traces":
        n = st.slider("Number of traces", 10, 200, 50)
        traces = tracer.recent(n)
        if traces:
            df = pd.DataFrame(traces)
            display_cols = ["timestamp", "strategy", "query", "confidence",
                            "latency_ms", "tokens_used"]
            st.dataframe(df[[c for c in display_cols if c in df.columns]],
                         use_container_width=True)

            # Click to expand full trace
            if st.checkbox("Show full trace JSON for latest"):
                st.json(traces[0])
        else:
            st.info("No traces yet. Run some queries first.")

    elif subtab == "Latency Summary":
        summary = tracer.strategy_latency_summary()
        if summary:
            df = pd.DataFrame(summary)
            st.dataframe(df, use_container_width=True)
            fig = px.bar(df, x="strategy", y="avg_latency_ms",
                         color="avg_confidence",
                         title="Avg Latency by Strategy",
                         color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet.")

    elif subtab == "Self-Healing Cases":
        cases = tracer.healing_cases()
        if cases:
            st.metric("Total healing events", len(cases))
            df = pd.DataFrame(cases)
            st.dataframe(
                df[["timestamp", "query", "strategy", "healing_chain", "confidence"]],
                use_container_width=True,
            )
        else:
            st.success(
                "No self-healing events — all queries met confidence threshold.")

# ------------------------------------------------------------------
# Tab 4: Knowledge Graph
# ------------------------------------------------------------------

elif tab == "Knowledge Graph":
    st.title("Knowledge Graph Explorer")
    kg = components["kg_store"]
    stats = kg.stats()

    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes", stats["nodes"])
    c2.metric("Edges", stats["edges"])
    c3.metric("Status", "Built" if not stats["is_empty"] else "Not built")

    if stats["is_empty"]:
        st.warning(
            "Knowledge graph is empty. Run ingestion with KG building enabled. "
            "Use: `python ingest.py --all --build-kg`"
        )
    else:
        entity_input = st.text_input(
            "Explore entity",
            placeholder="e.g. tp53, sql injection, kubernetes"
        )

        if entity_input:
            entity = entity_input.lower().strip()
            neighbours = list(kg.get_neighbors(entity, depth=2))
            relations = kg.get_relations(entity)

            st.subheader(f"Neighbourhood of '{entity}'")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{len(neighbours)} neighbours (depth=2)**")
                st.write(", ".join(neighbours[:30]))
                with st.expander("All edges (entity → neighbour)"):
                    for n in neighbours[:60]:
                        st.write(f"{entity} → {n}")
            with col2:
                st.write(f"**{len(relations)} direct relations**")
                rel_df = pd.DataFrame(relations)
                if not rel_df.empty:
                    rel_df["relations"] = rel_df["relations"].apply(
                        lambda r: ", ".join(r))
                    st.dataframe(rel_df, use_container_width=True)

            # Visualise with pyvis
            if st.button("Visualise Graph"):
                try:
                    from pyvis.network import Network

                    subgraph = kg.graph.subgraph(
                        [entity] + neighbours[:40]
                    )

                    net = Network(
                        height="500px",
                        width="100%",
                        bgcolor="#1e1e1e",
                        font_color="white",
                    )

                    net.from_nx(subgraph)

                    html = net.generate_html()

                    st.components.v1.html(
                        html,
                        height=520,
                    )

                except ImportError:
                    st.error(
                        "pyvis not installed. Run: pip install pyvis"
                    )
