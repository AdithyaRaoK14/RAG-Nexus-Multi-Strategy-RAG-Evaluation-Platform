from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional

from ingestion.loaders import DocumentLoader, RawDocument
from ingestion.chunkers import Chunker
from core.retriever import Retriever

logger = logging.getLogger(__name__)


class Indexer:
    """
    High-level ingestion orchestrator.

    Usage:
        indexer = Indexer(config, retriever)
        indexer.ingest_domain("medical", Path("./corpus/medical"))
        indexer.ingest_corpus(Path("./corpus"))
    """

    def __init__(self, config: dict, retriever: Retriever):
        self.config = config
        self.retriever = retriever
        self.loader = DocumentLoader()
        self.chunker = Chunker(config)

    def ingest_file(self, path: Path, domain: str) -> int:
        """Ingest a single file. Returns number of chunks indexed."""
        logger.info(f"Ingesting: {path.name} [{domain}]")
        docs = self.loader.load_file(path, domain)
        return self._index_docs(docs)

    def ingest_domain(self, domain: str, directory: Path) -> int:
        """Ingest all files in a domain directory."""
        logger.info(f"Ingesting domain '{domain}' from {directory}")
        docs = self.loader.load_directory(directory, domain)
        return self._index_docs(docs)

    def ingest_corpus(self, corpus_base: Optional[Path] = None) -> int:
        """Ingest the full multi-domain corpus."""
        base = corpus_base or Path(self.config["corpus"]["base_path"])
        logger.info(f"Ingesting full corpus from {base}")
        docs = self.loader.load_corpus(base)
        return self._index_docs(docs)

    def stats(self) -> dict:
        return self.retriever.collection_stats()

    # ------------------------------------------------------------------

    def _index_docs(self, docs: List[RawDocument]) -> int:
        if not docs:
            logger.warning("No documents found to index")
            return 0

        chunks = self.chunker.chunk_documents(docs)
        if not chunks:
            logger.warning("No chunks produced from documents")
            return 0

        chunk_dicts = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source": c.source,
                "domain": c.domain,
                "page": c.page,
            }
            for c in chunks
        ]

        self.retriever.index_chunks(chunk_dicts)
        logger.info(f"Indexed {len(chunk_dicts)} chunks")
        return len(chunk_dicts)
