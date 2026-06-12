from __future__ import annotations
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List

from core.schema import RAGResponse

logger = logging.getLogger(__name__)


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS traces (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    query           TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    query_type      TEXT,
    answer_preview  TEXT,
    confidence      REAL,
    latency_ms      REAL,
    tokens_used     INTEGER,
    router_scores   TEXT,   -- JSON
    reranker_scores TEXT,   -- JSON
    healing_chain   TEXT,   -- JSON list of fallback strategies
    retrieved_srcs  TEXT,   -- JSON list of source filenames
    extra           TEXT    -- JSON catch-all
);
"""


class Tracer:
    """
    Persists every RAG pipeline execution to SQLite.

    Streamlit dashboard queries this table to answer:
      - "Why did the router pick GraphRAG for this query?"
      - "How has latency changed across strategies over the last N runs?"
      - "Which queries triggered the self-healing fallback?"
    """

    def __init__(self, config: dict):
        self.enabled: bool = config.get("observability", {}).get("enabled", True)
        db_path = Path(config.get("observability", {}).get("db_path", "./data/traces.db"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self._init_db()

    def log(self, response: RAGResponse) -> None:
        """Persist a completed RAGResponse to the traces table."""
        if not self.enabled:
            return
        t = response.trace
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": response.query,
            "strategy": response.strategy,
            "query_type": t.query_type,
            "answer_preview": response.answer[:300],
            "confidence": response.confidence,
            "latency_ms": response.latency_ms,
            "tokens_used": t.tokens_estimated,
            "router_scores": json.dumps(t.router_scores),
            "reranker_scores": json.dumps(t.reranker_scores),
            "healing_chain": json.dumps(t.healing_attempts),
            "retrieved_srcs": json.dumps([c.source for c in response.sources]),
            "extra": json.dumps(t.extra),
        }
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO traces
                    (timestamp, query, strategy, query_type, answer_preview, confidence,
                     latency_ms, tokens_used, router_scores, reranker_scores,
                     healing_chain, retrieved_srcs, extra)
                    VALUES
                    (:timestamp, :query, :strategy, :query_type, :answer_preview, :confidence,
                     :latency_ms, :tokens_used, :router_scores, :reranker_scores,
                     :healing_chain, :retrieved_srcs, :extra)
                    """,
                    row,
                )
        except Exception as e:
            logger.error(f"Tracer write failed: {e}")

    # ------------------------------------------------------------------
    # Query helpers (used by Streamlit dashboard)
    # ------------------------------------------------------------------

    def recent(self, n: int = 50) -> List[dict]:
        """Return the n most recent traces."""
        return self._query(f"SELECT * FROM traces ORDER BY id DESC LIMIT {n}")

    def by_strategy(self, strategy: str, n: int = 100) -> List[dict]:
        return self._query(
            "SELECT * FROM traces WHERE strategy = ? ORDER BY id DESC LIMIT ?",
            (strategy, n),
        )

    def healing_cases(self) -> List[dict]:
        """Traces where the self-healing pipeline fired."""
        return self._query(
            "SELECT * FROM traces WHERE healing_chain != '[]' ORDER BY id DESC"
        )

    def strategy_latency_summary(self) -> List[dict]:
        return self._query(
            """
            SELECT strategy,
                   COUNT(*) as n,
                   ROUND(AVG(latency_ms), 1) as avg_latency_ms,
                   ROUND(AVG(confidence), 3) as avg_confidence
            FROM traces
            GROUP BY strategy
            ORDER BY avg_latency_ms
            """
        )

    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(CREATE_TABLE)

    def _query(self, sql: str, params: tuple = ()) -> List[dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Tracer query failed: {e}")
            return []
