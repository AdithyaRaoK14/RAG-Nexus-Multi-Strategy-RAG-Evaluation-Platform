from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import List, Set, Dict

import networkx as nx

from knowledge_graph.extractor import Triple

logger = logging.getLogger(__name__)


class KnowledgeGraphStore:
    """
    Persisted NetworkX DiGraph storing entity-relation-entity triples.

    Node attributes:  {frequency, domains, sources}
    Edge attributes:  {relation, weight, domains, sources}

    The graph is built automatically during document ingestion and
    queried by GraphRAG for neighbourhood-based retrieval.
    """

    def __init__(self, config: dict):
        self.store_path = Path(
            config.get("knowledge_graph", {}).get("store_path", "./data/knowledge_graph.pkl")
        )
        self.min_entity_freq: int = config.get("knowledge_graph", {}).get(
            "min_entity_freq", 2
        )
        self.graph: nx.DiGraph = nx.DiGraph()
        self._load()

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_triples(
        self,
        triples: List[Triple],
        source: str,
        domain: str,
    ) -> None:
        """Add a list of triples with provenance metadata."""
        for subj, rel, obj in triples:
            # Add / update nodes
            for entity in (subj, obj):
                if entity not in self.graph:
                    self.graph.add_node(entity, frequency=0, domains=[],sources=[])
                self.graph.nodes[entity]["frequency"] += 1
                if domain not in self.graph.nodes[entity]["domains"]:
                    self.graph.nodes[entity]["domains"].append(domain)
                if source not in self.graph.nodes[entity]["sources"]:
                    self.graph.nodes[entity]["sources"].append(source)

            # Add / update edge
            if self.graph.has_edge(subj, obj):
                self.graph[subj][obj]["weight"] += 1
                if rel not in self.graph[subj][obj]["relations"]:
                    self.graph[subj][obj]["relations"].append(rel)
                if domain not in self.graph[subj][obj]["domains"]:
                    self.graph[subj][obj]["domains"].append(domain)
                if source not in self.graph[subj][obj]["sources"]:
                    self.graph[subj][obj]["sources"].append(source)
            else:
                self.graph.add_edge(
                    subj, obj,
                    weight=1,
                    relations=[rel],
                    domains=[domain],
                    sources=[source],
                )

    def prune(self) -> None:
        """Remove low-frequency nodes (noise from bad extractions)."""
        to_remove = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("frequency", 0) < self.min_entity_freq
        ]
        self.graph.remove_nodes_from(to_remove)
        logger.info(
            f"Pruned {len(to_remove)} low-frequency nodes. "
            f"Graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    def save(self) -> None:

        self.store_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        graph_copy = self.graph.copy()

        for _, attrs in graph_copy.nodes(data=True):

            for key in ("domains", "sources"):

                if isinstance(attrs.get(key), set):

                    attrs[key] = list(attrs[key])

        for _, _, attrs in graph_copy.edges(data=True):

            for key in (
                "relations",
                "domains",
                "sources",
            ):

                if isinstance(attrs.get(key), set):

                    attrs[key] = list(attrs[key])

        with open(self.store_path, "wb") as f:

            pickle.dump(graph_copy, f)

        logger.info(
            f"Knowledge graph saved → {self.store_path}"
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        entity: str,
        depth: int = 2,
    ) -> Set[str]:
        """BFS expansion from entity up to `depth` hops."""
        if entity not in self.graph:
            return set()
        visited: Set[str] = {entity}
        frontier = {entity}
        for _ in range(depth):
            next_frontier: Set[str] = set()
            for node in frontier:
                next_frontier.update(self.graph.successors(node))
                next_frontier.update(self.graph.predecessors(node))
            next_frontier -= visited
            visited |= next_frontier
            frontier = next_frontier
        return visited

    def get_relations(self, entity: str) -> List[Dict]:
        """All edges (in + out) for an entity."""
        results = []
        for _, target, data in self.graph.out_edges(entity, data=True):
            results.append({
                "source": entity,
                "target": target,
                "relations": list(data.get("relations", [])),
                "weight": data.get("weight", 1),
            })
        for source, _, data in self.graph.in_edges(entity, data=True):
            results.append({
                "source": source,
                "target": entity,
                "relations": list(data.get("relations", [])),
                "weight": data.get("weight", 1),
            })
        return results

    def find_entities_in_text(self, text: str) -> List[str]:
        """Return graph entities that appear in the query text."""
        text_lower = text.lower()
        return [
            node for node in self.graph.nodes()
            if node in text_lower
        ]

    def subgraph_sources(self, entities: Set[str]) -> Set[str]:
        """All source documents that contributed to the given entity set."""
        sources: Set[str] = set()
        for entity in entities:
            if entity in self.graph:
                sources.update(
                    self.graph.nodes[entity].get(
                        "sources",
                        [],
                    )
                )
        return sources

    def stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "is_empty": self.graph.number_of_nodes() == 0,
        }

    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.store_path.exists():
            with open(self.store_path, "rb") as f:
                self.graph = pickle.load(f)

                for _, attrs in self.graph.nodes(data=True):

                    for key in ("domains", "sources"):

                        if isinstance(attrs.get(key), set):

                            attrs[key] = list(attrs[key])


                for _, _, attrs in self.graph.edges(data=True):

                    for key in (
                        "relations",
                        "domains",
                        "sources",
                    ):

                        if isinstance(attrs.get(key), set):

                            attrs[key] = list(attrs[key])
            logger.info(
                f"Knowledge graph loaded — "
                f"{self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges"
            )
