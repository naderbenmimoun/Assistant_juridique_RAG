"""Focused orchestration of rewrite, retrieval, and grounded generation."""

from dataclasses import dataclass
import json
import logging
import re
import time
from typing import Any, TypeVar, Sequence

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from ollama import ResponseError

from app.config import Settings
from app.rag.retriever import dynamic_filter, format_context, search_top_k, selected_source_cards, source_cards
from app.schemas import ChatMessage, ChatMetrics, ChatResponse, GeneratedAnswer, RouteDecision

logger = logging.getLogger(__name__)
ABSTENTION_ANSWER = "Le contexte fourni ne permet pas de répondre à cette question."
ROUTER_FALLBACK = "Je suis un assistant juridique. Je peux vous aider pour une question juridique."
StructuredOutput = TypeVar("StructuredOutput", RouteDecision, GeneratedAnswer)


def trim_history(history: Sequence[ChatMessage], max_turns: int) -> list[ChatMessage]:
    """Keep only recent browser-held turns; nothing is persisted server-side."""
    return list(history[-(max_turns * 2) :]) if max_turns else []


def to_langchain_history(history: Sequence[ChatMessage]) -> list[BaseMessage]:
    return [HumanMessage(item.content) if item.role == "user" else AIMessage(item.content) for item in history]


def remove_thinking(answer: str) -> str:
    """Defensively remove Qwen reasoning blocks; only final user-facing text remains."""
    if "<think" not in answer.lower():
        return answer.strip()
    match = re.search(r"</think>\s*(.*)$", answer, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_structured_output(raw: str, output_type: type[StructuredOutput]) -> StructuredOutput:
    """Parse the JSON-only LLM contract after removing any accidental reasoning."""
    cleaned = remove_thinking(raw).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return output_type.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("The model did not return a JSON object.")
        return output_type.model_validate(json.loads(match.group(0)))


@dataclass
class RagPipeline:
    vector_store: QdrantVectorStore
    embedder: OllamaEmbeddings
    router_chain: Any
    answer_chain: Any
    settings: Settings

    def chat(self, message: str, history: Sequence[ChatMessage]) -> ChatResponse:
        """Route normal conversation or run a grounded legal RAG answer."""
        started = time.perf_counter()
        recent = trim_history(history, self.settings.max_history_turns)
        lc_history = to_langchain_history(recent)
        router_started = time.perf_counter()
        try:
            route = parse_structured_output(
                str(self.router_chain.invoke({"history": lc_history, "message": message})), RouteDecision
            )
        except Exception:
            logger.warning("Router output was unusable; treating the message as a legal search.", exc_info=True)
            route = RouteDecision(intent="legal", search_query=message)
        router_ms = int((time.perf_counter() - router_started) * 1000)

        if route.intent != "legal":
            return ChatResponse(
                answer=remove_thinking(route.reply) or ROUTER_FALLBACK,
                standalone_query=message,
                sources=[],
                metrics=ChatMetrics(retrieved=0, retained=0, rewritten=False, embedding_ms=0,
                    retrieval_ms=0, generation_ms=router_ms, total_ms=int((time.perf_counter() - started) * 1000)),
            )
        standalone = route.search_query.strip()
        rewritten = standalone != message

        embed_started = time.perf_counter()
        results = search_top_k(
            self.vector_store, self.embedder, standalone, self.settings.retrieval_max_k, self.settings.vector_size
        )
        embedding_ms = int((time.perf_counter() - embed_started) * 1000)
        retrieval_started = time.perf_counter()
        retained = dynamic_filter(results, self.settings.retrieval_min_score, self.settings.retrieval_gap_cutoff)
        context = format_context(retained, self.settings.max_context_chars)
        retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)
        cards = source_cards(retained)

        if not retained:
            return ChatResponse(
                answer=ABSTENTION_ANSWER,
                standalone_query=standalone,
                sources=[],
                metrics=ChatMetrics(retrieved=len(results), retained=0, rewritten=rewritten,
                    embedding_ms=embedding_ms, retrieval_ms=retrieval_ms, generation_ms=0,
                    total_ms=int((time.perf_counter() - started) * 1000)),
            )

        generation_started = time.perf_counter()
        try:
            generated = parse_structured_output(
                str(self.answer_chain.invoke({"history": lc_history, "question": standalone, "context": context})),
                GeneratedAnswer,
            )
        except ResponseError as error:
            # Resource/model failures are dependency outages, not a reason to
            # claim that the legal corpus had insufficient context.
            logger.warning("Ollama generation is unavailable: %s", error)
            raise ConnectionError("Ollama generation is unavailable") from error
        except (httpx.TimeoutException, TimeoutError, ConnectionError, OSError):
            # The thin FastAPI route maps local dependency failures to 503/504.
            raise
        except Exception:
            logger.exception("Grounded answer generation failed")
            generated = None
        generation_ms = router_ms + int((time.perf_counter() - generation_started) * 1000)
        if generated is None or generated.answer == ABSTENTION_ANSWER:
            answer, selected_cards = ABSTENTION_ANSWER, []
        else:
            selected_cards = selected_source_cards(generated.used_chunk_ids, cards)
            if not selected_cards:
                answer, selected_cards = ABSTENTION_ANSWER, []
            else:
                answer = generated.answer
        return ChatResponse(
            answer=answer,
            standalone_query=standalone,
            sources=selected_cards,
            metrics=ChatMetrics(retrieved=len(results), retained=len(retained), rewritten=rewritten,
                embedding_ms=embedding_ms, retrieval_ms=retrieval_ms, generation_ms=generation_ms,
                total_ms=int((time.perf_counter() - started) * 1000)),
        )
