from __future__ import annotations
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    strategy: str = "adaptive_rag"
    domain: Optional[str] = None
    top_k: Optional[int] = None


class CompareRequest(BaseModel):
    query: str
    strategies: List[str] = ["naive_rag", "hybrid_rag", "advanced_rag"]
    domain: Optional[str] = None


def _get_state():
    from api.main import get_state
    return get_state()


@router.post("/")
def query(req: QueryRequest, state: dict = Depends(_get_state)):
    strategies = state.get("strategies", {})
    tracer = state.get("tracer")

    strat = strategies.get(req.strategy)
    if not strat:
        raise HTTPException(404, f"Strategy '{req.strategy}' not found. "
                                 f"Available: {list(strategies.keys())}")

    response = strat.run(query=req.query, domain_filter=req.domain, top_k=req.top_k)
    if tracer:
        tracer.log(response)
    return response.to_dict()


@router.post("/compare")
def compare(req: CompareRequest, state: dict = Depends(_get_state)):
    strategies = state.get("strategies", {})
    tracer = state.get("tracer")
    results = {}

    for name in req.strategies:
        strat = strategies.get(name)
        if not strat:
            results[name] = {"error": "strategy not found"}
            continue
        resp = strat.run(query=req.query, domain_filter=req.domain)
        if tracer:
            tracer.log(resp)
        results[name] = {
            "answer": resp.answer,
            "confidence": round(resp.confidence, 3),
            "latency_ms": round(resp.latency_ms, 1),
            "sources": [c.source for c in resp.sources[:3]],
        }
    return results


@router.get("/strategies")
def list_strategies(state: dict = Depends(_get_state)):
    return {"strategies": list(state.get("strategies", {}).keys())}
