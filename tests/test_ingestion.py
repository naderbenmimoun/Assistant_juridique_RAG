"""Small in-memory integration check for the LangChain Qdrant payload convention."""

from qdrant_client import QdrantClient, models
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore


class FixedEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def test_qdrant_payload_is_returned_as_langchain_document():
    client = QdrantClient(":memory:")
    client.create_collection("test_dole", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
    client.upsert("test_dole", points=[models.PointStruct(
        id=1, vector=[1.0, 0.0, 0.0],
        payload={"page_content": "Texte juridique complet", "metadata": {"chunk_id": "chunk-1", "title": "Titre"}},
    )])
    store = QdrantVectorStore(client=client, collection_name="test_dole", embedding=FixedEmbeddings(),
        content_payload_key="page_content", metadata_payload_key="metadata", vector_name="")
    document, _score = store.similarity_search_with_score("test", k=1)[0]
    assert document.page_content == "Texte juridique complet"
    assert document.metadata["chunk_id"] == "chunk-1"
    assert document.metadata["title"] == "Titre"
