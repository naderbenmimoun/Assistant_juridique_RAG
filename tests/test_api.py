from contextlib import asynccontextmanager

import httpx
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.rag.pipeline import ABSTENTION_ANSWER
from app.schemas import ChatMessage, ChatMetrics, ChatResponse, SourceCard


@asynccontextmanager
async def no_lifespan(app):
    yield


class FakePipeline:
    def chat(self, message, history):
        return ChatResponse(answer="Réponse [Source 1]", standalone_query=message, sources=[SourceCard(
            rank=1, chunk_id="chunk-1", doc_id="doc-1", title="Titre", score=0.7, excerpt="Extrait"
        )], metrics=ChatMetrics(retrieved=5, retained=1, rewritten=False, embedding_ms=0, retrieval_ms=0, generation_ms=0, total_ms=1))


class FakeQdrant:
    def count(self, *args, **kwargs):
        return type("Count", (), {"count": 4352})()


def test_chat_schema_and_input_validation():
    app = create_app(no_lifespan)
    app.state.settings = Settings()
    app.state.pipeline = FakePipeline()
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "Question", "history": []})
        assert response.status_code == 200
        assert response.json()["sources"][0]["chunk_id"] == "chunk-1"
        assert client.post("/api/chat", json={"message": "   "}).status_code == 422
        assert client.post("/api/chat", json={"message": "x" * 2001}).status_code == 422


def test_chat_timeout_maps_to_gateway_timeout():
    class TimeoutPipeline:
        def chat(self, message, history):
            raise TimeoutError("local model timeout")
    app = create_app(no_lifespan)
    app.state.settings = Settings()
    app.state.pipeline = TimeoutPipeline()
    with TestClient(app) as client:
        assert client.post("/api/chat", json={"message": "Question"}).status_code == 504


def test_health_healthy_and_unhealthy():
    app = create_app(no_lifespan)
    app.state.settings = Settings()
    app.state.qdrant = FakeQdrant()
    app.state.ollama_health_client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"models": [{"name": "qwen3:8b"}, {"name": "bge-m3"}]})
    ), base_url="http://test")
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
    app.state.ollama_health_client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(500)
    ), base_url="http://test")
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 503


def test_pipeline_abstains_without_generation_and_trims_history(monkeypatch):
    from app.rag.pipeline import RagPipeline, trim_history
    from app.schemas import ChatMessage

    class Embedder:
        def embed_query(self, query): return [0.0] * 1024
    class Store:
        def similarity_search_with_score(self, query, k): return []
    class LegalRouter:
        def invoke(self, value): return '{"intent":"legal","reply":"","search_query":"question"}'
    class NeverCalled:
        def invoke(self, value): raise AssertionError("generation must not run")
    pipeline = RagPipeline(Store(), Embedder(), LegalRouter(), NeverCalled(), Settings())
    response = pipeline.chat("question", [ChatMessage(role="user", content="a")] * 12)
    assert response.answer == ABSTENTION_ANSWER
    assert trim_history([ChatMessage(role="user", content="a")] * 10, 4) == [ChatMessage(role="user", content="a")] * 8


def test_router_failure_falls_back_to_original_legal_question():
    from app.rag.pipeline import RagPipeline

    class Embedder:
        def embed_query(self, query): return [0.0] * 1024
    class Store:
        def similarity_search_with_score(self, query, k): return []
    class FailingRouter:
        def invoke(self, value): raise RuntimeError("offline")
    pipeline = RagPipeline(Store(), Embedder(), FailingRouter(), FailingRouter(), Settings())
    response = pipeline.chat("Et si ?", [ChatMessage(role="user", content="Fraude")])
    assert response.standalone_query == "Et si ?" and response.metrics.rewritten is False


def test_router_handles_conversation_without_retrieval():
    from app.rag.pipeline import RagPipeline

    class NeverCalled:
        def __getattr__(self, name): raise AssertionError("legal RAG must not run for conversation")
    class ConversationRouter:
        def invoke(self, value):
            return '{"intent":"conversational","reply":"Bonjour ! Comment puis-je vous aider ?","search_query":""}'
    response = RagPipeline(NeverCalled(), NeverCalled(), ConversationRouter(), NeverCalled(), Settings()).chat("bonjour", [])
    assert response.sources == [] and response.metrics.retrieved == 0
    assert "Bonjour" in response.answer


def test_thinking_content_is_not_exposed():
    from app.rag.pipeline import remove_thinking
    assert remove_thinking("<think>secret reasoning</think>Réponse finale") == "Réponse finale"


def test_only_valid_model_selected_chunk_ids_are_exposed():
    from langchain_core.documents import Document
    from app.rag.pipeline import RagPipeline

    class Embedder:
        def embed_query(self, query): return [0.0] * 1024
    class Store:
        def similarity_search_with_score(self, query, k):
            return [(Document(page_content="Texte", metadata={"chunk_id": "chunk-1", "doc_id": "doc-1"}), 0.7)]
    class Router:
        def invoke(self, value): return '{"intent":"legal","reply":"","search_query":"question juridique"}'
    class Answer:
        def invoke(self, value):
            return '{"answer":"Réponse fondée.","used_chunk_ids":["chunk-1","invented"]}'
    response = RagPipeline(Store(), Embedder(), Router(), Answer(), Settings()).chat("question", [])
    assert response.answer == "Réponse fondée."
    assert [source.chunk_id for source in response.sources] == ["chunk-1"]
