from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from rank_bm25 import BM25Okapi

from core.schema import RetrievedChunk
from core.embedder import Embedder

logger = logging.getLogger(__name__)

BM25_INDEX_PATH = Path("./data/bm25_index.pkl")
BM25_CORPUS_PATH = Path("./data/bm25_corpus.pkl")


class Retriever:
    """
    Hybrid retriever combining:
      - Dense search via Qdrant (BAAI/bge-base-en-v1.5)
      - Sparse search via BM25Okapi
      - Reciprocal Rank Fusion (RRF) for result merging
    """

    def __init__(self, config: dict, embedder: Embedder):
        self.config = config
        self.embedder = embedder
        self.cfg_qdrant = config["qdrant"]
        self.cfg_ret = config["retrieval"]
        self.collection = self.cfg_qdrant["collection_name"]

        # Qdrant client
        if self.cfg_qdrant.get("mode", "local") == "server":
            host = self.cfg_qdrant.get("host", "localhost")
            port = self.cfg_qdrant.get("port", 6333)
            self.qdrant = QdrantClient(host=host, port=port)
            logger.info(f"Qdrant client → server @ {host}:{port}")
        else:
            local_path = self.cfg_qdrant.get("local_path", "./data/qdrant")
            Path(local_path).mkdir(parents=True, exist_ok=True)
            self.qdrant = QdrantClient(path=local_path)
            logger.info(f"Qdrant client → local @ {local_path}")

        self._ensure_collection()

        # BM25 index (loaded from disk if exists)
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_docs: List[dict] = []      # parallel list of chunk metadata
        self.bm25_texts: List[str] = []      # raw texts for BM25
        self._load_bm25()

    # ------------------------------------------------------------------
    # Retrieval entrypoints
    # ------------------------------------------------------------------

    def dense_search(
        self,
        query: str,
        top_k: int = 10,
        domain_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Pure dense vector search via Qdrant."""
        query_vec = self.embedder.embed_query(query).tolist()

        qfilter = None
        if domain_filter:
            qfilter = Filter(
                must=[FieldCondition(key="domain", match=MatchValue(value=domain_filter))]
            )

        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        )

        return [self._scored_point_to_chunk(r, sparse_score=0.0) for r in results]

    def sparse_search(
        self,
        query: str,
        top_k: int = 10,
        domain_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """BM25 sparse search."""
        if self.bm25 is None:
            logger.warning("BM25 index not built yet — returning empty")
            return []

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Apply domain filter
        if domain_filter:
            for i, doc in enumerate(self.bm25_docs):
                if doc.get("domain") != domain_filter:
                    scores[i] = 0.0

        top_indices = np.argsort(scores)[::-1][:top_k]

        chunks = []
        max_score = float(scores[top_indices[0]]) if len(top_indices) > 0 else 1.0
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc = self.bm25_docs[idx]
            chunks.append(RetrievedChunk(
                chunk_id=doc["chunk_id"],
                text=self.bm25_texts[idx],
                source=doc["source"],
                domain=doc["domain"],
                page=doc["page"],
                dense_score=0.0,
                sparse_score=round(float(scores[idx]) / max(max_score, 1e-9), 4),
            ))
        return chunks

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        domain_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Hybrid search: dense + sparse, fused with RRF.
        Each strategy's top candidates are merged; duplicates are resolved
        by taking the higher individual scores.
        """
        dense_results = self.dense_search(query, top_k=top_k, domain_filter=domain_filter)
        sparse_results = self.sparse_search(query, top_k=top_k, domain_filter=domain_filter)

        fused_scores = self._rrf_fusion(
            [
                [c.chunk_id for c in dense_results],
                [c.chunk_id for c in sparse_results],
            ]
        )

        # Build a lookup of chunk_id → chunk object
        chunk_map: Dict[str, RetrievedChunk] = {}
        for c in dense_results:
            chunk_map[c.chunk_id] = c
        for c in sparse_results:
            if c.chunk_id in chunk_map:
                chunk_map[c.chunk_id].sparse_score = c.sparse_score
            else:
                chunk_map[c.chunk_id] = c

        # Sort by RRF score and return top_k
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [chunk_map[cid] for cid, _ in ranked if cid in chunk_map]

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def index_chunks(self, chunks: List[dict]) -> None:
        """
        Index a list of chunk dicts into Qdrant + BM25.

        Expected dict shape:
          {
            "chunk_id": str,
            "text": str,
            "source": str,
            "domain": str,
            "page": int,
            "embedding": np.ndarray  (optional, will compute if missing)
          }
        """
        # Compute embeddings for chunks that don't have them
        texts_to_embed = [c["text"] for c in chunks if "embedding" not in c]
        if texts_to_embed:
            logger.info(f"Embedding {len(texts_to_embed)} chunks...")
            embeddings = self.embedder.embed_documents(texts_to_embed)
            idx = 0
            for c in chunks:
                if "embedding" not in c:
                    c["embedding"] = embeddings[idx]
                    idx += 1

        # Upsert into Qdrant
        logger.info(f"Upserting {len(chunks)} chunks into Qdrant...")
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            points = [
                PointStruct(
                    id=self._chunk_id_to_int(c["chunk_id"]),
                    vector=c["embedding"].tolist(),
                    payload={
                        "chunk_id": c["chunk_id"],
                        "source": c["source"],
                        "domain": c["domain"],
                        "page": c["page"],
                        "text": c["text"],
                    },
                )
                for c in batch
            ]
            self.qdrant.upsert(collection_name=self.collection, points=points)

        # Update BM25 index
        logger.info("Rebuilding BM25 index...")
        for c in chunks:
            self.bm25_docs.append({
                "chunk_id": c["chunk_id"],
                "source": c["source"],
                "domain": c["domain"],
                "page": c["page"],
            })
            self.bm25_texts.append(c["text"])

        tokenized = [self._tokenize(t) for t in self.bm25_texts]
        self.bm25 = BM25Okapi(tokenized)
        self._save_bm25()
        logger.info(f"Indexed {len(chunks)} chunks successfully")

    def collection_stats(self) -> dict:
        try:
            count = self.qdrant.count(
                collection_name=self.collection,
                exact=True,
            )

            qdrant_vectors = count.count

        except Exception:
            qdrant_vectors = 0

        return {
            "qdrant_vectors": qdrant_vectors,
            "bm25_documents": len(self.bm25_docs),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection not in existing:
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.cfg_qdrant.get("vector_size", 768),
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection}")

    def _rrf_fusion(
        self,
        rankings: List[List[str]],
        k: int = None,
    ) -> Dict[str, float]:
        """Reciprocal Rank Fusion."""
        k = k or self.cfg_ret.get("rrf_k", 60)
        scores: Dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return scores

    def _scored_point_to_chunk(self, point, sparse_score: float = 0.0) -> RetrievedChunk:
        p = point.payload
        return RetrievedChunk(
            chunk_id=p.get("chunk_id", str(point.id)),
            text=p.get("text", ""),
            source=p.get("source", "unknown"),
            domain=p.get("domain", "unknown"),
            page=p.get("page", 0),
            dense_score=round(float(point.score), 4),
            sparse_score=sparse_score,
        )

    def _chunk_id_to_int(self, chunk_id: str) -> int:
        """Stable int ID from chunk_id string via hash."""
        return abs(hash(chunk_id)) % (2**31)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Basic tokenizer for BM25."""
        import re
        text = text.lower()
        tokens = re.findall(r"\b[a-z][a-z0-9]*\b", text)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "or",
            "and", "but", "not", "this", "that", "it", "its",
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def _save_bm25(self) -> None:
        BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(BM25_CORPUS_PATH, "wb") as f:
            pickle.dump({"docs": self.bm25_docs, "texts": self.bm25_texts}, f)

    def _load_bm25(self) -> None:
        if BM25_INDEX_PATH.exists() and BM25_CORPUS_PATH.exists():
            with open(BM25_INDEX_PATH, "rb") as f:
                self.bm25 = pickle.load(f)
            with open(BM25_CORPUS_PATH, "rb") as f:
                data = pickle.load(f)
                self.bm25_docs = data["docs"]
                self.bm25_texts = data["texts"]
            logger.info(f"BM25 index loaded — {len(self.bm25_docs)} documents")
