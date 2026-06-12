from __future__ import annotations
import logging
import re
from typing import Optional, List, TypedDict, Literal

from langgraph.graph import StateGraph, END

from strategies.base import BaseRAGStrategy
from core.schema import RAGResponse, RAGTrace, RetrievedChunk

logger = logging.getLogger(__name__)

MAX_STEPS = 6

# -----------------------------------------------------------------------
# ReAct prompt
# The agent reasons about which of RAG-Nexus's own tools to use.
# Works with 3B models via strict format enforcement + robust parsing.
# -----------------------------------------------------------------------

REACT_SYSTEM = """\
You are a research agent with access to a local knowledge base.
Answer the user's question by reasoning step-by-step and using tools.

AVAILABLE TOOLS (call only these):
- dense_search(query)   : semantic similarity search over document embeddings
- hybrid_search(query)  : dense + BM25 keyword search combined (best general)
- graph_search(entity)  : look up entity relationships in the knowledge graph
- hyde_search(query)    : generate a hypothetical answer then retrieve by it

RULES:
- Use 2–4 tool calls before writing Final Answer
- Each tool call must use a DIFFERENT query or entity from the previous
- If one tool gives poor results, try another with a refined query

RESPONSE FORMAT — follow exactly:
Thought: <reason about what information you still need>
Action: <tool_name>
Action Input: <input string>

When you have enough information:
Thought: I have enough information to write a complete answer.
Final Answer: <comprehensive answer based on retrieved information>
"""

REACT_USER = """\
Question: {query}

{scratchpad}
Observation from last tool:
{last_observation}

Continue:"""


# -----------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    domain_filter: Optional[str]
    scratchpad: str
    last_observation: str
    all_chunks: List[dict]          # serialised RetrievedChunk dicts
    step_count: int
    final_answer: str
    done: bool
    _next_action: Optional[str]     # set by think node, consumed by execute node
    _next_input: Optional[str]


class AgenticRAG(BaseRAGStrategy):
    """
    ReAct-loop RAG agent.

    The agent decides WHICH retrieval tool to call and with WHAT query,
    can call tools multiple times, accumulates evidence, then synthesises
    a final answer. Unlike AdaptiveRAG (classify-once-then-route), this
    agent genuinely reasons across multiple retrieval steps.

    Tools available:
      dense_search   — Qdrant semantic search
      hybrid_search  — Dense + BM25 + RRF (default workhorse)
      graph_search   — KG neighbourhood expansion
      hyde_search    — Hypothetical document embedding search

    All tools are RAG-Nexus's own; no external APIs used.
    """

    def __init__(self, config, retriever, reranker, generator, kg_store=None):
        super().__init__(config, retriever, reranker, generator)
        self.kg_store = kg_store
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # BaseRAGStrategy interface
    # ------------------------------------------------------------------

    def retrieve_and_generate(
        self,
        query: str,
        trace: RAGTrace,
        domain_filter: Optional[str],
        top_k: int,
    ) -> RAGResponse:

        initial: AgentState = {
            "query": query,
            "domain_filter": domain_filter,
            "scratchpad": "",
            "last_observation": "No tool called yet. Start with a tool call.",
            "all_chunks": [],
            "step_count": 0,
            "final_answer": "",
            "done": False,
            "_next_action": None,
            "_next_input": None,
        }

        final = self._graph.invoke(initial)

        # Deserialise + rerank all accumulated chunks
        raw_chunks = final["all_chunks"]
        chunks = [self._dict_to_chunk(c) for c in raw_chunks]
        if chunks:
            chunks = self.reranker.rerank(query, chunks, top_k=self.cfg_ret["rerank_top_k"])

        trace.extra["agent_steps"] = final["step_count"]
        trace.extra["scratchpad"] = final["scratchpad"]
        trace.reranker_scores = [round(c.rerank_score, 4) for c in chunks if c.rerank_score]
        confidence = self._score_confidence(query, chunks)
        trace.confidence = confidence
        trace.tokens_estimated = self.generator.estimate_tokens(final["scratchpad"])

        answer = final["final_answer"] or self._fallback_generate(query, chunks)

        return RAGResponse(
            query=query,
            answer=answer,
            sources=chunks,
            strategy="agentic_rag",
            trace=trace,
            latency_ms=0.0,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------

    def _think_act_node(self, state: AgentState) -> dict:
        """LLM reasons about the next tool call."""
        prompt = REACT_USER.format(
            query=state["query"],
            scratchpad=state["scratchpad"],
            last_observation=state["last_observation"],
        )
        raw, _ = self.generator.generate(prompt, system=REACT_SYSTEM)

        # Check for final answer first
        if "Final Answer:" in raw:
            answer = raw.split("Final Answer:", 1)[1].strip()
            new_scratchpad = state["scratchpad"] + "\n" + raw
            return {
                "final_answer": answer,
                "scratchpad": new_scratchpad,
                "done": True,
                "step_count": state["step_count"] + 1,
            }

        # Parse Action / Action Input
        action, action_input = self._parse_action(raw)
        new_scratchpad = state["scratchpad"] + "\n" + raw
        return {
            "scratchpad": new_scratchpad,
            "step_count": state["step_count"] + 1,
            "_next_action": action,
            "_next_input": action_input,
            "done": False,
        }

    def _execute_tool_node(self, state: AgentState) -> dict:
        """Execute the tool the agent chose."""
        action = state.get("_next_action", "hybrid_search")
        action_input = state.get("_next_input", state["query"])
        domain = state["domain_filter"]

        chunks: List[RetrievedChunk] = []
        observation = ""

        try:
            if action == "dense_search":
                chunks = self.retriever.dense_search(action_input, top_k=6, domain_filter=domain)
            elif action == "graph_search":
                chunks = self._graph_search(action_input, domain)
                observation = self._kg_observation(action_input)
            elif action == "hyde_search":
                chunks = self._hyde_search(action_input, domain)
            else:  # hybrid_search (default)
                chunks = self.retriever.hybrid_search(action_input, top_k=6, domain_filter=domain)

            if not observation:
                if chunks:
                    tops = chunks[:3]
                    observation = f"Retrieved {len(chunks)} chunks. Top results:\n"
                    for i, c in enumerate(tops):
                        observation += f"[{i+1}] ({c.source}, score={c.final_score:.3f}): {c.text[:200]}...\n"
                else:
                    observation = "No relevant documents found. Try a different query."
        except Exception as e:
            observation = f"Tool error: {e}. Try a different approach."
            logger.warning(f"AgenticRAG tool error ({action}): {e}")

        # Merge new chunks (dedup by chunk_id)
        existing_ids = {c["chunk_id"] for c in state["all_chunks"]}
        new_chunks = [
            self._chunk_to_dict(c) for c in chunks if c.chunk_id not in existing_ids
        ]

        return {
            "all_chunks": state["all_chunks"] + new_chunks,
            "last_observation": observation,
            "_next_action": None,
            "_next_input": None,
        }

    def _should_continue(self, state: AgentState) -> Literal["act", "done"]:
        if state["done"] or state["step_count"] >= MAX_STEPS:
            return "done"
        return "act"

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("think", self._think_act_node)
        builder.add_node("execute", self._execute_tool_node)

        builder.set_entry_point("think")
        builder.add_conditional_edges(
            "think",
            self._should_continue,
            {"act": "execute", "done": END},
        )
        builder.add_edge("execute", "think")
        return builder.compile()

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _hyde_search(self, query: str, domain: Optional[str]) -> List[RetrievedChunk]:
        hyde_prompt = self.generator.build_hyde_prompt(query)
        hyde_doc, _ = self.generator.generate(hyde_prompt)
        hyde_vec = self.retriever.embedder.embed_documents([hyde_doc])[0]
        raw = self.retriever.qdrant.search(
            collection_name=self.retriever.collection,
            query_vector=hyde_vec.tolist(),
            limit=6,
            with_payload=True,
        )
        return [self.retriever._scored_point_to_chunk(r) for r in raw]

    def _graph_search(self, entity: str, domain: Optional[str]) -> List[RetrievedChunk]:
        if not self.kg_store or self.kg_store.stats()["is_empty"]:
            return self.retriever.hybrid_search(entity, top_k=6, domain_filter=domain)
        neighbours = self.kg_store.get_neighbors(entity.lower(), depth=2)
        sources = self.kg_store.subgraph_sources(neighbours | {entity.lower()})
        candidates = self.retriever.hybrid_search(entity, top_k=10, domain_filter=domain)
        for c in candidates:
            if c.source in sources:
                c.dense_score = min(1.0, c.dense_score * 1.2)
        return sorted(candidates, key=lambda c: c.dense_score, reverse=True)[:6]

    def _kg_observation(self, entity: str) -> str:
        if not self.kg_store or self.kg_store.stats()["is_empty"]:
            return "Knowledge graph not yet built."
        rels = self.kg_store.get_relations(entity.lower())[:5]
        if not rels:
            return f"No graph entries for '{entity}'. Returning hybrid search results."
        lines = [f"  {r['source']} --[{', '.join(r['relations'])}]--> {r['target']}" for r in rels]
        return "Knowledge graph relations:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Fallback + parsing helpers
    # ------------------------------------------------------------------

    def _fallback_generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "Insufficient context retrieved to answer this question."
        context = self._format_context(chunks[:5])
        prompt = self.generator.build_rag_prompt(query, context)
        answer, _ = self.generator.generate(prompt)
        return answer

    @staticmethod
    def _parse_action(raw: str):
        action_match = re.search(r"Action\s*:\s*(\w+)", raw)
        input_match = re.search(r"Action Input\s*:\s*(.+?)(?:\n|$)", raw)
        action = action_match.group(1).strip().lower() if action_match else "hybrid_search"
        action_input = input_match.group(1).strip() if input_match else ""
        # Normalise tool names
        tool_map = {
            "dense": "dense_search", "hybrid": "hybrid_search",
            "graph": "graph_search", "hyde": "hyde_search",
        }
        for short, full in tool_map.items():
            if action.startswith(short):
                action = full
                break
        return action, action_input

    @staticmethod
    def _chunk_to_dict(c: RetrievedChunk) -> dict:
        return {
            "chunk_id": c.chunk_id, "text": c.text, "source": c.source,
            "domain": c.domain, "page": c.page,
            "dense_score": c.dense_score, "sparse_score": c.sparse_score,
        }

    @staticmethod
    def _dict_to_chunk(d: dict) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=d["chunk_id"], text=d["text"], source=d["source"],
            domain=d["domain"], page=d["page"],
            dense_score=d["dense_score"], sparse_score=d["sparse_score"],
        )
