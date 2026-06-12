from __future__ import annotations
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter()


class IngestRequest(BaseModel):
    domain: str
    path: str


class IngestCorpusRequest(BaseModel):
    corpus_path: Optional[str] = None


def _get_state():
    from api.main import get_state
    return get_state()


@router.post("/file")
def ingest_domain(req: IngestRequest, state: dict = Depends(_get_state)):
    indexer = state.get("indexer")
    if not indexer:
        raise HTTPException(500, "Indexer not initialised")
    n = indexer.ingest_domain(req.domain, Path(req.path))
    return {"indexed_chunks": n, "domain": req.domain}


@router.post("/corpus")
def ingest_corpus(req: IngestCorpusRequest, state: dict = Depends(_get_state)):
    indexer = state.get("indexer")
    if not indexer:
        raise HTTPException(500, "Indexer not initialised")
    path = Path(req.corpus_path) if req.corpus_path else None
    n = indexer.ingest_corpus(path)
    return {"indexed_chunks": n}
