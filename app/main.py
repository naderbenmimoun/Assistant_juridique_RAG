"""FastAPI wiring for the local legal RAG application."""

from contextlib import asynccontextmanager
import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator, Callable

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from app.config import Settings, get_settings
from app.ingestion import EXPECTED_POINTS, ensure_collection, wait_for_qdrant
from app.rag.chains import build_answer_chain, build_router_chain
from app.rag.pipeline import RagPipeline
from app.schemas import ChatRequest, ChatResponse, ErrorResponse, HealthResponse

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


def _has_ollama_model(model_names: set[str], configured_name: str) -> bool:
    """Accept Ollama's displayed ``:latest`` tag for an untagged configured model."""
    return configured_name in model_names or (
        ":" not in configured_name and f"{configured_name}:latest" in model_names
    )


async def _wait_for_ollama(client: httpx.AsyncClient, attempts: int = 20) -> None:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = await client.get("/api/tags")
            response.raise_for_status()
            models_present = {model.get("name") for model in response.json().get("models", [])}
            settings = get_settings()
            if _has_ollama_model(models_present, settings.ollama_chat_model) and _has_ollama_model(
                models_present, settings.ollama_embedding_model
            ):
                return
            last_error = RuntimeError("Configured Ollama models are not pulled yet.")
        except Exception as error:
            last_error = error
        await asyncio.sleep(min(0.5 * (2 ** min(attempt, 4)), 5))
    raise RuntimeError("Ollama or its configured models are unavailable after startup retries.") from last_error


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    qdrant = QdrantClient(url=settings.qdrant_url, timeout=30)
    health_client = httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=httpx.Timeout(10.0, read=10.0))
    try:
        await asyncio.to_thread(wait_for_qdrant, qdrant)
        await _wait_for_ollama(health_client)
        if settings.auto_ingest:
            await asyncio.to_thread(ensure_collection, qdrant, settings)
        embeddings = OllamaEmbeddings(model=settings.ollama_embedding_model, base_url=settings.ollama_base_url)
        chat_model = ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
            num_ctx=settings.ollama_num_ctx,
            num_predict=settings.ollama_num_predict,
            format="json",
            extract_reasoning=True,
        )
        vector_store = QdrantVectorStore(
            client=qdrant,
            collection_name=settings.qdrant_collection,
            embedding=embeddings,
            content_payload_key="page_content",
            metadata_payload_key="metadata",
            vector_name="",
            distance=models.Distance.COSINE,
        )
        app.state.settings = settings
        app.state.qdrant = qdrant
        app.state.ollama_health_client = health_client
        app.state.pipeline = RagPipeline(vector_store, embeddings, build_router_chain(chat_model), build_answer_chain(chat_model), settings)
        yield
    finally:
        await health_client.aclose()
        qdrant.close()


def get_pipeline(request: Request) -> RagPipeline:
    """Small dependency boundary for the lifespan-created shared pipeline."""
    return request.app.state.pipeline


def create_app(lifespan_handler: Callable | None = None) -> FastAPI:
    app = FastAPI(title="Assistant juridique", lifespan=lifespan_handler or lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def home() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health", response_model=HealthResponse, responses={503: {"model": ErrorResponse}})
    async def health(request: Request) -> HealthResponse:
        settings: Settings = request.app.state.settings
        try:
            count = await asyncio.to_thread(request.app.state.qdrant.count, settings.qdrant_collection, exact=True)
            ollama_response = await request.app.state.ollama_health_client.get("/api/tags")
            ollama_response.raise_for_status()
            model_names = {model.get("name") for model in ollama_response.json().get("models", [])}
            if (
                count.count != EXPECTED_POINTS
                or not _has_ollama_model(model_names, settings.ollama_chat_model)
                or not _has_ollama_model(model_names, settings.ollama_embedding_model)
            ):
                raise RuntimeError("A required dependency is not ready.")
            return HealthResponse(status="ok", qdrant="ok", ollama="ok", embedding_model=settings.ollama_embedding_model,
                collection=settings.qdrant_collection, points=count.count, llm_model=settings.ollama_chat_model)
        except Exception:
            logger.warning("Health check dependency failure", exc_info=True)
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Une dépendance locale est indisponible.")

    @app.post("/api/chat", response_model=ChatResponse, responses={503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}})
    async def chat(payload: ChatRequest, request: Request, pipeline: RagPipeline = Depends(get_pipeline)) -> ChatResponse:
        settings: Settings = request.app.state.settings
        if len(payload.message) > settings.max_question_chars:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Le message dépasse la longueur autorisée.")
        try:
            result = await asyncio.to_thread(pipeline.chat, payload.message, payload.history)
            logger.info("Chat completed: retained=%s duration_ms=%s", result.metrics.retained, result.metrics.total_ms)
            return result
        except (httpx.TimeoutException, TimeoutError):
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Le modèle local a expiré. Réessayez.")
        except (ConnectionError, OSError):
            logger.warning("Local RAG dependency unavailable", exc_info=True)
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Une dépendance locale est indisponible.")
        except Exception:
            logger.exception("Unexpected chat failure")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Une erreur inattendue est survenue.")

    return app


app = create_app()
