from __future__ import annotations
import hashlib
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional
from collections.abc import Generator as StreamGenerator
import requests

logger = logging.getLogger(__name__)


class Generator:
    """
    Thin wrapper around Ollama's REST API.

    Keeps zero LangChain dependencies so strategies can call it directly
    without the LangChain abstraction overhead.
    """

    def __init__(self, config: dict):
        cfg = config["models"]["llm"]
        self.model: str = cfg["default"]
        self.base_url: str = cfg.get("base_url", "http://localhost:11434")
        self.temperature: float = cfg.get("temperature", 0.1)
        self.max_tokens: int = cfg.get("max_tokens", 1024)
        self.timeout: int = cfg.get("timeout", 120)
        self._generate_url = f"{self.base_url}/api/generate"
        self._chat_url = f"{self.base_url}/api/chat"
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        self._cache_db = cache_dir / "ollama_cache.sqlite"
        self._init_cache()
        logger.info(f"Generator ready — model={self.model} @ {self.base_url}")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system: Optional[str] = None) -> tuple[str, float]:
        """
        Generate a response. Returns (answer_text, latency_ms).
        Uses /api/generate with a single prompt string.
        Retries once if Ollama returns a transient HTTP error.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        if system:
            payload["system"] = system

        cache_key = hashlib.sha256(
            (
                self.model
                + prompt
                + str(system)
                + str(self.temperature)
                + str(self.max_tokens)
            ).encode()
        ).hexdigest()

        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached, 0.0

        t0 = time.perf_counter()

        for attempt in range(2):
            try:
                resp = requests.post(
                    self._generate_url,
                    json=payload,
                    timeout=self.timeout,
                )

                resp.raise_for_status()

                latency_ms = (time.perf_counter() - t0) * 1000
                data = resp.json()
                answer = data.get("response", "").strip()

                self._cache_set(cache_key, answer)

                return answer, latency_ms

            except requests.exceptions.ConnectionError:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Run: ollama serve"
                )

            except requests.exceptions.Timeout:
                if attempt == 0:
                    logger.warning(
                        f"Ollama request timed out after {self.timeout}s. "
                        "Retrying in 5 seconds..."
                    )
                    time.sleep(5)
                    continue

                raise RuntimeError(
                    f"Ollama request timed out after {self.timeout} seconds."
                )

            except requests.exceptions.HTTPError as e:
                status_code = (
                    e.response.status_code
                    if e.response is not None
                    else 0
                )

                if attempt == 0 and status_code >= 500:
                    logger.warning(
                        f"Ollama returned HTTP {status_code}. "
                        "Retrying in 5 seconds..."
                    )
                    time.sleep(5)
                    continue

                raise RuntimeError(
                    f"Ollama generation failed with HTTP {status_code}: "
                    f"{e.response.text if e.response is not None else str(e)}"
                )

        raise RuntimeError("Generation failed after retries.")

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> StreamGenerator[str, None, None]:
        """Streaming generate — yields tokens as they arrive."""
        import json

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system:
            payload["system"] = system

        with requests.post(
            self._generate_url,
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("response", "")
                    if chunk.get("done"):
                        break

    # ------------------------------------------------------------------
    # Prompt templates
    # ------------------------------------------------------------------

    def build_rag_prompt(self, query: str, context_chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(
            f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
        return (
            f"You are a precise research assistant. Answer the question using ONLY "
            f"the provided context. If the context does not contain enough information, "
            f"say so explicitly. Do not fabricate.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:"
        )

    def build_rewrite_prompt(self, query: str) -> str:
        return (
            f"Rewrite the following question to be more specific and retrieval-friendly. "
            f"Output ONLY the rewritten question, nothing else.\n\n"
            f"Original: {query}\n\nRewritten:"
        )

    def build_hyde_prompt(self, query: str) -> str:
        return (
            f"Write a short, factual paragraph (3-5 sentences) that would directly answer "
            f"the following question. This will be used for document retrieval.\n\n"
            f"Question: {query}\n\nParagraph:"
        )

    def build_faithfulness_prompt(self, query: str, answer: str, context: str) -> str:
        return (
            f"Given the question and retrieved context below, score how faithfully the answer "
            f"is supported by the context. Return ONLY a float between 0.0 and 1.0.\n\n"
            f"Question: {query}\n\n"
            f"Context: {context[:1500]}\n\n"
            f"Answer: {answer}\n\n"
            f"Faithfulness score (0.0 to 1.0):"
        )

    def build_claim_judge_prompt(self, answer: str, context: str) -> str:
        return f"""
    You are evaluating whether an answer is supported by evidence.

    CONTEXT:
    {context[:3000]}

    ANSWER:
    {answer}

    Extract atomic factual claims from the ANSWER.

    For each claim assign exactly one label:

    SUPPORTED
    PARTIALLY_SUPPORTED
    UNSUPPORTED

    Return ONLY VALID JSON.

    Rules:
    1. Output must be a JSON array.
    2. Use double quotes around ALL property names.
    3. Use double quotes around ALL string values.
    4. Do NOT include explanations.
    5. Do NOT include markdown fences.
    6. Output MUST be parseable by Python json.loads().

    Example output:

    [
        {{
            "claim": "Word embeddings improve semantic matching.",
            "label": "SUPPORTED"
        }},
        {{
            "claim": "The method achieved 99% accuracy.",
            "label": "UNSUPPORTED"
        }}
    ]

    JSON:
    """

    def build_confidence_prompt(self, query: str, context_chunks: list[str]) -> str:
        context = "\n".join(c[:300] for c in context_chunks[:3])
        return (
            f"Does the following context contain sufficient information to answer the question? "
            f"Return ONLY a float between 0.0 and 1.0 where 1.0 = fully sufficient.\n\n"
            f"Question: {query}\nContext: {context}\n\nScore:"
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _init_cache(self) -> None:
        with sqlite3.connect(self._cache_db) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    ts    INTEGER DEFAULT (strftime('%s','now'))
                )
                """
            )

    def _cache_get(self, key: str) -> Optional[str]:
        with sqlite3.connect(self._cache_db) as con:
            row = con.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def _cache_set(self, key: str, value: str) -> None:
        with sqlite3.connect(self._cache_db) as con:
            con.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, value),
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
