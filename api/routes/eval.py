from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter()


class BenchmarkRequest(BaseModel):
    benchmark_file: str
    domain_filter: Optional[str] = None


def _get_state():
    from api.main import get_state
    return get_state()


@router.post("/benchmark")
def run_benchmark(req: BenchmarkRequest, state: dict = Depends(_get_state)):
    runner = state.get("runner")
    if not runner:
        raise HTTPException(500, "Runner not initialised")
    results = runner.run(req.benchmark_file, domain_filter=req.domain_filter)
    df = runner.aggregate_table(results)
    return df.to_dict(orient="records")


@router.get("/traces")
def get_traces(n: int = 50, state: dict = Depends(_get_state)):
    tracer = state.get("tracer")
    if not tracer:
        raise HTTPException(500, "Tracer not initialised")
    return tracer.recent(n)


@router.get("/traces/healing")
def get_healing_cases(state: dict = Depends(_get_state)):
    tracer = state.get("tracer")
    return tracer.healing_cases() if tracer else []


@router.get("/traces/summary")
def get_strategy_summary(state: dict = Depends(_get_state)):
    tracer = state.get("tracer")
    return tracer.strategy_latency_summary() if tracer else []
