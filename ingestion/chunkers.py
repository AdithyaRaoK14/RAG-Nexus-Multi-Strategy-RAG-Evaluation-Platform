from __future__ import annotations
import re
import hashlib
import logging
from typing import List
from dataclasses import dataclass

from ingestion.loaders import RawDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    domain: str
    page: int
    chunk_index: int   # position within the source document


class Chunker:
    """
    Splits RawDocuments into Chunks.

    Strategies:
      - recursive: respects paragraph / sentence boundaries (default)
      - fixed: hard token-count splits with overlap
    """

    def __init__(self, config: dict):
        cfg = config["chunking"]
        self.strategy: str = cfg.get("strategy", "recursive")
        self.chunk_size: int = cfg.get("chunk_size", 512)
        self.chunk_overlap: int = cfg.get("chunk_overlap", 64)
        self.min_chunk_size: int = cfg.get("min_chunk_size", 100)

    def chunk_documents(self, documents: List[RawDocument]) -> List[Chunk]:
        """Chunk a list of RawDocuments."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)
        logger.info(
            f"Chunked {len(documents)} pages → {len(all_chunks)} chunks "
            f"(strategy={self.strategy}, size={self.chunk_size})"
        )
        return all_chunks

    def _chunk_document(self, doc: RawDocument) -> List[Chunk]:
        if self.strategy == "fixed":
            raw_chunks = self._fixed_split(doc.text)
        else:
            raw_chunks = self._recursive_split(doc.text)

        chunks = []
        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if len(text) < self.min_chunk_size:
                continue
            chunk_id = self._make_chunk_id(doc.source, doc.page, i)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=text,
                source=doc.source,
                domain=doc.domain,
                page=doc.page,
                chunk_index=i,
            ))
        return chunks

    # ------------------------------------------------------------------
    # Splitting strategies
    # ------------------------------------------------------------------

    def _recursive_split(self, text: str) -> List[str]:
        """
        Tries to split on paragraph → sentence → word boundaries.
        Respects chunk_size in approximate character counts.
        """
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
        return self._recursive_split_inner(text, separators)

    def _recursive_split_inner(self, text: str, separators: List[str]) -> List[str]:
        if not separators or len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = separators[0]
        parts = text.split(sep)

        chunks = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    # Recurse with finer separator
                    sub = self._recursive_split_inner(part, separators[1:])
                    chunks.extend(sub)
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current)

        # Apply overlap: prepend tail of previous chunk to next
        if self.chunk_overlap > 0:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _fixed_split(self, text: str) -> List[str]:
        """Hard split by character count with overlap."""
        words = text.split()
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)

        i = 0
        while i < len(words):
            window = words[i : i + self.chunk_size]
            chunks.append(" ".join(window))
            i += step

        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.chunk_overlap :] if self.chunk_overlap else ""
            result.append(tail + " " + chunks[i] if tail else chunks[i])
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_chunk_id(source: str, page: int, index: int) -> str:
        raw = f"{source}|p{page}|c{index}"
        digest = hashlib.md5(raw.encode()).hexdigest()[:8]
        # Keep it human-readable: stem_p3_c1_a1b2c3d4
        stem = re.sub(r"[^a-zA-Z0-9]", "_", source.replace(".pdf", ""))[:20]
        return f"{stem}_p{page}_c{index}_{digest}"
