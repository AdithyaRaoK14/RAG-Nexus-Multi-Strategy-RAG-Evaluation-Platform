from __future__ import annotations
import logging
from pathlib import Path
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    text: str
    source: str       # filename
    domain: str       # from corpus directory name
    page: int = 0


class DocumentLoader:
    """
    Loads PDF, TXT and Markdown files from a directory tree.

    Expected corpus layout:
        corpus/
          medical/scc/paper_01.pdf
          security/sqli/paper_01.pdf
          ...

    The domain is inferred from the top-level subdirectory name.
    """

    SUPPORTED = {".pdf", ".txt", ".md"}

    def load_file(self, path: Path, domain: str) -> List[RawDocument]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(path, domain)
        elif suffix in {".txt", ".md"}:
            return self._load_text(path, domain)
        else:
            logger.warning(f"Unsupported file type: {path}")
            return []

    def load_directory(self, directory: Path, domain: str) -> List[RawDocument]:
        """Recursively load all supported files from a directory."""
        docs: List[RawDocument] = []
        files = [p for p in directory.rglob("*") if p.suffix.lower() in self.SUPPORTED]
        logger.info(f"Found {len(files)} files in {directory}")
        for f in files:
            docs.extend(self.load_file(f, domain))
        return docs

    def load_corpus(self, corpus_base: Path) -> List[RawDocument]:
        """Load the full multi-domain corpus."""
        all_docs: List[RawDocument] = []
        for domain_dir in sorted(corpus_base.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name
            docs = self.load_directory(domain_dir, domain)
            logger.info(f"Domain '{domain}': loaded {len(docs)} pages")
            all_docs.extend(docs)
        logger.info(f"Total corpus: {len(all_docs)} pages across all domains")
        return all_docs

    # ------------------------------------------------------------------
    # Format-specific loaders
    # ------------------------------------------------------------------

    def _load_pdf(self, path: Path, domain: str) -> List[RawDocument]:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = self._clean_text(text)
                if len(text) > 50:   # skip near-empty pages
                    docs.append(RawDocument(
                        text=text,
                        source=path.name,
                        domain=domain,
                        page=i + 1,
                    ))
            logger.debug(f"Loaded {len(docs)} pages from {path.name}")
            return docs
        except Exception as e:
            logger.error(f"Failed to load PDF {path}: {e}")
            return []

    def _load_text(self, path: Path, domain: str) -> List[RawDocument]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            text = self._clean_text(text)
            if not text.strip():
                return []
            return [RawDocument(text=text, source=path.name, domain=domain, page=1)]
        except Exception as e:
            logger.error(f"Failed to load text file {path}: {e}")
            return []

    @staticmethod
    def _clean_text(text: str) -> str:
        import re
        # Remove excessive whitespace and control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()
