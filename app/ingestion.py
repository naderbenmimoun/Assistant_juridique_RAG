"""Idempotent import of the experiment's existing BGE-M3 vectors into Qdrant."""

import argparse
import ast
import json
import logging
import math
import time
import uuid
from typing import Any, Iterable

from qdrant_client import QdrantClient, models

from app.config import Settings, get_settings

EXPECTED_POINTS = 4352
logger = logging.getLogger(__name__)


def _repair_command() -> str:
    return "docker compose run --rm api python -m app.ingestion --recreate"


def wait_for_qdrant(client: QdrantClient, attempts: int = 20) -> None:
    """Use short bounded retries because Compose readiness alone is insufficient."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            client.get_collections()
            return
        except Exception as error:
            last_error = error
            time.sleep(min(0.5 * (2 ** min(attempt, 4)), 5))
    raise RuntimeError("Qdrant is unavailable after startup retries.") from last_error


def parse_embedding(raw: Any, vector_size: int) -> list[float]:
    """Parse stored embeddings without ever recomputing corpus vectors."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = ast.literal_eval(raw)
    vector = [float(value) for value in raw]
    if len(vector) != vector_size or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"Dataset contains an invalid embedding; expected {vector_size} finite values.")
    return vector


def _collection_size(client: QdrantClient, name: str) -> int | None:
    if not client.collection_exists(name):
        return None
    info = client.get_collection(name)
    vectors = info.config.params.vectors
    size = vectors.size if isinstance(vectors, models.VectorParams) else None
    if size != 1024:
        raise RuntimeError(f"Collection '{name}' has an incompatible vector size. Run: {_repair_command()}")
    return client.count(name, exact=True).count


def _create_collection(client: QdrantClient, settings: Settings) -> None:
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(size=settings.vector_size, distance=models.Distance.COSINE),
    )


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    def text(name: str) -> str:
        value = row.get(name)
        return "" if value is None else str(value)

    return {
        "page_content": text("chunk_text"),
        "metadata": {
            "chunk_id": text("chunk_id"), "doc_id": text("doc_id"), "title": text("title"),
            "category": text("category"), "content_type": text("content_type"),
            "chunk_index": row.get("chunk_index"), "creation_date": text("creation_date"),
            "article_number": text("article_number"),
        },
    }


def _batches(items: Iterable[models.PointStruct], size: int) -> Iterable[list[models.PointStruct]]:
    batch: list[models.PointStruct] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def ensure_collection(client: QdrantClient, settings: Settings, recreate: bool = False) -> None:
    """Import exactly 4,352 normalized existing vectors, or leave a healthy collection alone."""
    wait_for_qdrant(client)
    existing = _collection_size(client, settings.qdrant_collection)
    if recreate and existing is not None:
        client.delete_collection(settings.qdrant_collection)
        existing = None
    if existing == EXPECTED_POINTS:
        logger.info("Qdrant collection '%s' is already complete; skipping ingestion.", settings.qdrant_collection)
        return
    if existing not in (None, 0):
        raise RuntimeError(
            f"Collection '{settings.qdrant_collection}' contains {existing}, not {EXPECTED_POINTS}, points. "
            f"It will not be overwritten automatically. Run: {_repair_command()}"
        )
    if existing == 0:
        client.delete_collection(settings.qdrant_collection)
    _create_collection(client, settings)

    from datasets import load_dataset

    dataset = load_dataset(settings.hf_dataset, split=settings.hf_dataset_split)
    required = {"chunk_text", "embeddings_bge-m3", "chunk_id", "doc_id", "title", "category", "content_type", "chunk_index", "creation_date", "article_number"}
    missing = required.difference(dataset.column_names)
    if missing or len(dataset) != EXPECTED_POINTS:
        raise RuntimeError(f"Unexpected AgentPublic/dole dataset shape (missing={sorted(missing)}, rows={len(dataset)}).")

    points = (
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"AgentPublic/dole/{row['chunk_id']}")),
            vector=parse_embedding(row["embeddings_bge-m3"], settings.vector_size),
            payload=_payload(row),
        )
        for row in dataset
    )
    imported = 0
    for batch in _batches(points, 256):
        client.upsert(settings.qdrant_collection, points=batch, wait=True)
        imported += len(batch)
        logger.info("Imported %s/%s vectors", imported, EXPECTED_POINTS)
    final_count = client.count(settings.qdrant_collection, exact=True).count
    if final_count != EXPECTED_POINTS:
        raise RuntimeError(f"Import incomplete ({final_count}/{EXPECTED_POINTS}). Run: {_repair_command()}")
    logger.info("Imported %s existing BGE-M3 vectors into Qdrant.", final_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import AgentPublic/dole's stored embeddings into Qdrant.")
    parser.add_argument("--recreate", action="store_true", help="Intentionally recreate only the named collection.")
    args = parser.parse_args()
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, timeout=30)
    ensure_collection(client, settings, recreate=args.recreate)


if __name__ == "__main__":
    main()
