"""Explicit score-aware Qdrant retrieval and deterministic source rendering."""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

from app.schemas import SourceCard


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    score: float


def validate_query_vector(vector: Sequence[float], vector_size: int) -> list[float]:
    """Check the BGE-M3 runtime contract before a vector reaches Qdrant."""
    values = [float(value) for value in vector]
    if len(values) != vector_size or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Invalid query embedding: expected {vector_size} finite values.")
    return values


def dynamic_filter(
    results: Sequence[ScoredDocument], min_score: float = 0.40, gap_cutoff: float = 0.04
) -> list[ScoredDocument]:
    """Apply the notebook-validated absolute threshold and score-gap Dynamic-K rule."""
    if not results or results[0].score < min_score:
        return []
    kept = [results[0]]
    for previous, current in zip(results, results[1:]):
        if current.score < min_score or previous.score - current.score >= gap_cutoff:
            break
        kept.append(current)
    return kept


def search_top_k(
    vector_store: QdrantVectorStore,
    embedder: OllamaEmbeddings,
    query: str,
    max_k: int,
    vector_size: int,
) -> list[ScoredDocument]:
    """Embed once, validate it, then issue a score-aware Qdrant top-K query.

    The dataset vectors are normalized and Qdrant uses cosine distance, so higher
    returned scores are more similar and retain the notebook's inner-product ranking.
    """
    # Validate the embedding contract explicitly before asking LangChain/Qdrant to
    # run its score-aware query. The installed dedicated integration exposes the
    # query-string variant (which invokes its configured embedding function).
    validate_query_vector(embedder.embed_query(query), vector_size)
    pairs = vector_store.similarity_search_with_score(query, k=max_k)
    scored = [ScoredDocument(document=document, score=float(score)) for document, score in pairs]
    return sorted(scored, key=lambda item: item.score, reverse=True)


def source_cards(results: Iterable[ScoredDocument], excerpt_chars: int = 700) -> list[SourceCard]:
    """Map trusted Qdrant payload metadata into browser-safe source cards."""
    cards: list[SourceCard] = []
    for rank, result in enumerate(results, start=1):
        metadata = result.document.metadata
        excerpt = result.document.page_content.strip()
        if len(excerpt) > excerpt_chars:
            excerpt = excerpt[:excerpt_chars].rstrip() + "…"
        cards.append(
            SourceCard(
                rank=rank,
                chunk_id=str(metadata.get("chunk_id", "")),
                doc_id=str(metadata.get("doc_id", "")),
                title=str(metadata.get("title") or ""),
                category=str(metadata.get("category") or ""),
                content_type=str(metadata.get("content_type") or ""),
                article_number=str(metadata.get("article_number") or ""),
                creation_date=str(metadata.get("creation_date") or ""),
                score=round(result.score, 4),
                excerpt=excerpt,
            )
        )
    return cards


def selected_source_cards(chunk_ids: Sequence[str], cards: Sequence[SourceCard]) -> list[SourceCard]:
    """Keep only LLM-selected IDs that are trusted retained Qdrant sources.

    The model can choose from IDs it saw in context, but cannot introduce source
    metadata or arbitrary IDs. Retrieval rank remains the display order.
    """
    selected_ids = set(chunk_ids)
    return [card for card in cards if card.chunk_id in selected_ids]


def format_context(results: Iterable[ScoredDocument], max_chars: int) -> str:
    """Format retained chunks in rank order without silently reordering evidence."""
    parts: list[str] = []
    used = 0
    for rank, result in enumerate(results, start=1):
        metadata = result.document.metadata
        block = (
            f"[Source {rank}]\n"
            f"chunk_id: {metadata.get('chunk_id', '')}\n"
            f"Titre: {metadata.get('title', '')}\n"
            f"Article: {metadata.get('article_number', '')}\n"
            f"Catégorie: {metadata.get('category', '')}\n"
            f"Type: {metadata.get('content_type', '')}\n"
            f"Date: {metadata.get('creation_date', '')}\n"
            f"Contenu:\n{result.document.page_content.strip()}"
        )
        separator = "\n\n" if parts else ""
        remaining = max_chars - used - len(separator)
        if remaining <= 0:
            break
        if len(block) > remaining:
            if not parts:
                parts.append(block[: max(0, remaining - 24)].rstrip() + "\n[Contenu tronqué]")
            break
        parts.append(block)
        used += len(separator) + len(block)
    return "\n\n".join(parts)
