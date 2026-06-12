from __future__ import annotations
import json
import logging
import re
from typing import List

from core.schema import RAGResponse, BenchmarkQuery, BenchmarkResult
from core.generator import Generator

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Computes retrieval and generation quality metrics.

    Retrieval metrics (require ground-truth expected_docs):
      - Hit@1, Hit@3, Hit@5   — was a relevant doc in the top-k?
      - MRR                   — mean reciprocal rank of first relevant doc

    Generation metrics (LLM-as-judge via Ollama):
      - Faithfulness          — is the answer grounded in the retrieved context?
      - Context precision     — how much of the context was actually used?
    """

    def __init__(self, generator: Generator):
        self.generator = generator

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def evaluate(
        self,
        response: RAGResponse,
        benchmark_query: BenchmarkQuery,
    ) -> BenchmarkResult:
        retrieved_sources = list(
            dict.fromkeys(
                c.source for c in response.sources
            )
        )

        return BenchmarkResult(
            query=response.query,
            strategy=response.strategy,
            hit_at_1=self.hit_at_k(
                retrieved_sources, benchmark_query.expected_docs, k=1),
            hit_at_3=self.hit_at_k(
                retrieved_sources, benchmark_query.expected_docs, k=3),
            hit_at_5=self.hit_at_k(
                retrieved_sources, benchmark_query.expected_docs, k=5),
            mrr=self.mrr(retrieved_sources, benchmark_query.expected_docs),
            faithfulness=self.faithfulness(
                response.query,
                response.answer,
                [c.text for c in response.sources],
            ),
            context_precision=self.context_precision(
                retrieved_sources,
                benchmark_query.expected_docs,
            ),
            latency_ms=response.latency_ms,
            retrieved_sources=retrieved_sources,
            recall_at_5=self.recall_at_k(
                retrieved_sources, benchmark_query.expected_docs, k=5),
            recall_at_10=self.recall_at_k(
                retrieved_sources, benchmark_query.expected_docs, k=10),
        )

    # ------------------------------------------------------------------
    # Retrieval metrics
    # ------------------------------------------------------------------

    @staticmethod
    def hit_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
        """1 if any expected doc appears in retrieved[:k], else 0."""
        expected_set = {e.lower() for e in expected}
        for src in retrieved[:k]:
            if src.lower() in expected_set:
                return 1.0
        return 0.0

    @staticmethod
    def mrr(retrieved: List[str], expected: List[str]) -> float:
        """Reciprocal rank of the first relevant document."""
        expected_set = {e.lower() for e in expected}
        for rank, src in enumerate(retrieved, start=1):
            if src.lower() in expected_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        expected: List[str],
        k: int,
    ) -> float:
        """Fraction of expected docs found in retrieved[:k]."""

        if not expected:
            return 0.0

        expected_set = {
            e.lower()
            for e in expected
        }

        retrieved_set = {
            src.lower()
            for src in retrieved[:k]
        }

        hits = len(
            retrieved_set.intersection(expected_set)
        )

        return round(
            hits / len(expected_set),
            4,
        )

    @staticmethod
    def context_precision(retrieved: List[str], expected: List[str]) -> float:
        """
        Fraction of retrieved documents that are relevant.
        = precision@k where k = len(retrieved)
        """
        if not retrieved:
            return 0.0
        expected_set = {e.lower() for e in expected}
        relevant = sum(1 for src in retrieved if src.lower() in expected_set)
        return relevant / len(retrieved)

    # ------------------------------------------------------------------
    # Generation metrics (LLM-as-judge)
    # ------------------------------------------------------------------

    def faithfulness(
        self,
        query: str,
        answer: str,
        context_chunks: List[str],
        max_context_chars: int = 2000,
    ) -> float:
        """
        Claim-level faithfulness: extract atomic claims from the answer,
        classify each as SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED,
        return fraction of fully supported claims.
        """
        if not answer or not context_chunks:
            return 0.0

        context = "\n\n".join(context_chunks)[:max_context_chars]
        prompt = self.generator.build_claim_judge_prompt(answer, context)

        try:
            raw, _ = self.generator.generate(prompt)
            # strip markdown fences if model wraps output
            clean = re.sub(
                            r"```(?:json)?|```",
                            "",
                            raw,
                        ).strip()


            match = re.search(
                r"\[.*\]",
                clean,
                re.DOTALL,
            )

            if match:
                clean = match.group()

            claims = json.loads(clean)
            if not claims:
                return 0.0
            score = 0.0


            for c in claims:
                label = c.get("label")

                if label == "SUPPORTED":
                    score += 1.0

                elif label == "PARTIALLY_SUPPORTED":
                    score += 0.5

            return round(
                score / len(claims),
                4,
            )
        except Exception as e:
            logger.warning(
                f"Claim-level faithfulness failed: {e}"
            )

            return 0.5

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate(results: List[BenchmarkResult]) -> dict:
        """Compute mean metrics across a list of BenchmarkResults."""

        if not results:
            return {}

        keys = [
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "mrr",
            "faithfulness",
            "context_precision",
            "latency_ms",
            "recall_at_5",
            "recall_at_10",
        ]

        agg = {}

        for k in keys:

            vals = [
                getattr(r, k)
                for r in results
                if getattr(r, k) is not None
            ]

            agg[k] = (
                round(sum(vals) / len(vals), 4)
                if vals
                else 0.0
            )

        agg["n"] = len(results)

        return agg

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_float(text: str) -> float:
        """Extract first float from a string like '0.87' or 'Score: 0.87'."""
        matches = re.findall(r"0?\.\d+|\d+\.\d+|\d+", text.strip())
        if matches:
            return float(matches[0])
        return 0.5   # neutral fallback
