from __future__ import annotations
import ast
import json
import logging
import re
from typing import List, Tuple

from core.generator import Generator

logger = logging.getLogger(__name__)

# (subject, relation, object)
Triple = Tuple[str, str, str]

# Fix 1: BAD_ENTITIES moved to module level (outside the loop)
BAD_ENTITIES = {
    "true",
    "false",
    "system",
    "response",
    "input",
    "output",
    "text",
    "result",
    "results",
    "model",
    "models",
    "prompt",
    "prompts",
    "experiment",
    "experiments",
    "participant",
    "participants",
    "vector",
    "vectors",
    "5",
    "10",
}


EXTRACTION_PROMPT = """
Extract entity relationship triples from the text.

Return ONLY valid JSON.

Format:

[
  {{
    "head": "entity",
    "relation": "relationship",
    "tail": "entity"
  }}
]

Rules:
- Use ONLY the keys: "head", "relation", and "tail".
- Return ONLY JSON.
- Do NOT use markdown fences.
- Do NOT include explanations.
- Extract factual relationships between important domain concepts.
- Ignore generic words such as "system", "response", "input", "output", "true", "false", and standalone numbers.
- Prefer domain-specific entities.
- Maximum triples: {max_triples}

Text:
{text}
"""


class EntityExtractor:
    """
    Uses a local Ollama model to extract triples from document chunks.

    Triple format:
        (subject, relation, object)
    """

    def __init__(self, config: dict, generator: Generator):
        self.generator = generator

        self.max_triples = (
            config.get("knowledge_graph", {})
            .get("max_triples_per_chunk", 8)
        )

    def extract(self, text: str) -> List[Triple]:
        """Extract triples from a single chunk."""

        prompt = EXTRACTION_PROMPT.format(
            text=text[:1200],
            max_triples=self.max_triples,
        )

        try:
            raw, _ = self.generator.generate(prompt)

            return self._parse(raw)

        except Exception as e:
            logger.warning(
                f"Triple extraction failed: {e}"
            )
            return []

    def extract_batch(
        self,
        texts: List[str],
    ) -> List[List[Triple]]:
        """Extract triples from multiple chunks."""

        results = []

        for i, text in enumerate(texts):

            triples = self.extract(text)

            results.append(triples)

            if (i + 1) % 20 == 0:
                logger.info(
                    f"Extracted triples from "
                    f"{i + 1}/{len(texts)} chunks"
                )

        return results

    def _normalize_relation(self, relation: str) -> str:
        """Map noisy LLM relations into a small vocabulary."""

        relation = relation.lower().strip()

        relation = relation.replace("-", " ")
        relation = relation.replace("_", " ")

        mapping = {
            "is a major risk factor for": "risk_factor_for",
            "is a risk factor for": "risk_factor_for",
            "major risk factor for": "risk_factor_for",
            "is risk factor for": "risk_factor_for",
            "risk factor for": "risk_factor_for",

            "associated with": "associated_with",
            "is associated with": "associated_with",
            "are associated with": "associated_with",
            "frequently observed in": "associated_with",
            "is frequently observed in": "associated_with",
            "are frequently observed in": "associated_with",
            "observed in": "associated_with",

            "uses": "uses",
            "utilizes": "uses",
            "employs": "uses",

            "improves": "improves",
            "improve": "improves",
            "improved": "improves",
            "improves performance of": "improves",
            "enhances": "improves",
            "enhances performance of": "improves",
            "optimizes": "improves",

            "causes": "causes",
            "leads to": "causes",
            "results in": "causes",

            "implements": "implements",
            "implemented with": "implements",
            "based on": "implements",

            "detects": "detects",
            "identifies": "detects",
            "classifies": "detects",

            "part of": "part_of",
            "belongs to": "part_of",
        }

        return mapping.get(relation, "related_to")

    def _parse(self, raw: str) -> List[Triple]:

        raw = raw.strip()

        # Remove markdown fences
        raw = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        )

        raw = re.sub(
            r"\s*```$",
            "",
            raw,
        )

        raw = raw.strip()

        if raw == "[]":
            return []

        # Find outermost JSON array
        start = raw.find("[")
        end = raw.rfind("]")

        if start == -1 or end == -1 or end <= start:

            logger.debug(
                f"No JSON array found:\n{raw[:500]}"
            )

            return []

        json_text = raw[start:end + 1]

        repaired = json_text

        try:

            data = json.loads(json_text)

        except json.JSONDecodeError:

            # Fix unquoted property names:
            # { head: "x" } -> { "head": "x" }

            repaired = re.sub(
                r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:',
                r'\1"\2":',
                json_text,
            )

            try:

                data = json.loads(repaired)

                logger.info(
                    "Recovered malformed JSON using regex repair"
                )

            except json.JSONDecodeError:

                try:

                    data = ast.literal_eval(repaired)

                    logger.info(
                        "Recovered malformed JSON using ast.literal_eval"
                    )

                except Exception as e:

                    logger.warning(
                        f"Failed to parse JSON: {e}"
                    )

                    logger.warning(
                        f"Bad JSON output:\n{raw[:1000]}"
                    )

                    return []

        triples: List[Triple] = []

        for item in data:

            subject = ""
            relation = ""
            obj = ""

            # Dictionary outputs
            if isinstance(item, dict):

                subject = str(
                    item.get("subject")
                    or item.get("head")
                    or ""
                ).strip().lower()

                relation = str(
                    item.get("relation")
                    or item.get("predicate")
                    or ""
                ).strip().lower()

                obj = str(
                    item.get("object")
                    or item.get("tail")
                    or ""
                ).strip().lower()

            # List outputs
            elif (
                isinstance(item, list)
                and len(item) == 3
            ):

                subject = str(item[0]).strip().lower()

                relation = str(item[1]).strip().lower()

                obj = str(item[2]).strip().lower()

            else:

                continue

            relation = self._normalize_relation(relation)

            # Fix 2: Normalize whitespace BEFORE the self-loop check
            subject = re.sub(
                r"\s+",
                " ",
                subject,
            ).strip()

            obj = re.sub(
                r"\s+",
                " ",
                obj,
            ).strip()

            if relation == "related_to" and subject == obj:
                continue

            # Validation
            if (
                subject
                and relation
                and obj
                and len(subject) > 2
                and len(obj) > 2
                and len(subject) < 100
                and len(obj) < 100
            ):
                # Fix 3: Uses module-level BAD_ENTITIES (no re-definition inside loop)
                if subject in BAD_ENTITIES:
                    continue

                if obj in BAD_ENTITIES:
                    continue

                triples.append(
                    (
                        subject,
                        relation,
                        obj,
                    )
                )

        # Remove duplicates while preserving order
        triples = list(dict.fromkeys(triples))

        return triples
