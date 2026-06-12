#!/usr/bin/env python3
"""
RAG-Nexus ingestion CLI

Usage:
  python ingest.py --all                            # ingest full corpus
  python ingest.py --domain medical --path ./corpus/medical
  python ingest.py --all --build-kg                 # ingest + build knowledge graph
  python ingest.py --kg-only                        # rebuild KG only (skips Qdrant + BM25)
  python ingest.py --stats
"""
import logging
from pathlib import Path
from typing import Optional
import re
import typer
import yaml
from rich.console import Console

app = typer.Typer(help="RAG-Nexus document ingestion")
console = Console()


def _load(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logging.basicConfig(
        level=config["logging"]["level"], format=config["logging"]["format"])
    from core.embedder import Embedder
    from core.retriever import Retriever
    from core.generator import Generator
    from ingestion.indexers import Indexer
    from knowledge_graph.extractor import EntityExtractor
    from knowledge_graph.graph_store import KnowledgeGraphStore

    embedder = Embedder(config)
    retriever = Retriever(config, embedder)
    generator = Generator(config)
    indexer = Indexer(config, retriever)
    extractor = EntityExtractor(config, generator)
    kg_store = KnowledgeGraphStore(config)
    return {
        "config": config, "indexer": indexer, "retriever": retriever,
        "extractor": extractor, "kg_store": kg_store,
    }


@app.command()
def ingest(
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    path: Optional[Path] = typer.Option(None, "--path", "-p"),
    all_domains: bool = typer.Option(False, "--all", "-a"),
    build_kg: bool = typer.Option(
        False, "--build-kg", help="Extract entities + build KG"),
    kg_only: bool = typer.Option(
        False, "--kg-only", help="Rebuild knowledge graph only (skips Qdrant + BM25)"),
    stats: bool = typer.Option(False, "--stats", "-s"),
    config_path: str = typer.Option("config.yaml", "--config"),
):
    state = _load(config_path)
    indexer = state["indexer"]
    console.print(
        f"DEBUG: stats={stats}, "
        f"all_domains={all_domains}, "
        f"build_kg={build_kg}, "
        f"kg_only={kg_only}, "
        f"domain={domain}, "
        f"path={path}"
    )

    if stats:
        s = indexer.stats()
        kg_stats = state["kg_store"].stats()
        console.print("\n[bold]Index stats[/bold]")
        console.print(f"  Qdrant vectors : {s.get('qdrant_vectors', 0)}")
        console.print(f"  BM25 documents : {s.get('bm25_documents', 0)}")
        console.print("\n[bold]Knowledge graph[/bold]")
        console.print(f"  Nodes : {kg_stats['nodes']}")
        console.print(f"  Edges : {kg_stats['edges']}\n")
        return

    if kg_only:
        base = Path(state["config"]["corpus"]["base_path"])
        console.print(
            "[bold]Rebuilding knowledge graph only[/bold] (Qdrant + BM25 untouched)")
        _build_knowledge_graph(state, base)
        return

    if all_domains:
        base = Path(state["config"]["corpus"]["base_path"])
        console.print(f"[bold]Ingesting corpus from[/bold] {base}")
        n = indexer.ingest_corpus(base)
        console.print(f"[green]✓ Indexed {n} chunks[/green]")
        if build_kg:
            _build_knowledge_graph(state, base)
    elif domain and path:
        console.print(f"[bold]Ingesting[/bold] {domain} from {path}")
        n = indexer.ingest_domain(domain, path)
        console.print(f"[green]✓ Indexed {n} chunks[/green]")
        if build_kg:
            _build_knowledge_graph(state, path, domain=domain)
    else:
        console.print(
            "[red]Provide --domain + --path, or --all, or --stats[/red]")
        raise typer.Exit(1)


def _build_knowledge_graph(state: dict, base_path: Path, domain: Optional[str] = None):
    from ingestion.loaders import DocumentLoader
    from ingestion.chunkers import Chunker

    loader = DocumentLoader()
    chunker = Chunker(state["config"])
    extractor = state["extractor"]
    kg_store = state["kg_store"]
    kg_store.graph.clear()

    if domain:
        docs = loader.load_directory(base_path, domain)
    else:
        docs = loader.load_corpus(base_path)

    console.print(
        f"\n[bold]Building knowledge graph from {len(docs)} pages...[/bold]")
    chunks = chunker.chunk_documents(docs)

    # Extract every 10th chunk (~10% of corpus) to balance quality and speed
    sample = chunks[::10]
    console.print(f"Extracting triples from {len(sample)} chunks (sampled)...")

    total_triples = 0
    for i, chunk in enumerate(sample):
        text = chunk.text

        if len(text.split()) < 40:
            continue

        if len(re.findall(r"\[\d+\]", text)) > 5:
            continue

        if re.search(r"\b(FIGURE|TABLE)\s+\d+\b", text, re.IGNORECASE):
            continue
        triples = extractor.extract(text)
        if triples:
            kg_store.add_triples(
                triples, source=chunk.source, domain=chunk.domain)
            total_triples += len(triples)
        if (i + 1) % 50 == 0:
            console.print(
                f"  {i+1}/{len(sample)} chunks — {total_triples} triples so far")

    kg_store.prune()
    kg_store.save()
    stats = kg_store.stats()
    console.print(
        f"[green]✓ Knowledge graph built — "
        f"{stats['nodes']} nodes, {stats['edges']} edges[/green]"
    )


if __name__ == "__main__":
    app()
