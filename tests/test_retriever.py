from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.rag.chains import build_answer_chain, build_router_chain
from app.rag.prompts import ANSWER_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT
from app.rag.retriever import ScoredDocument, dynamic_filter, format_context, selected_source_cards, source_cards


def scored(scores):
    return [ScoredDocument(Document(page_content=f"texte {index}", metadata={"chunk_id": str(index)}), score) for index, score in enumerate(scores)]


def test_dynamic_filter_notebook_cases():
    assert dynamic_filter([]) == []
    assert dynamic_filter(scored([0.39])) == []
    assert len(dynamic_filter(scored([0.66, 0.64, 0.61]))) == 3
    assert len(dynamic_filter(scored([0.66, 0.61, 0.60]))) == 1
    assert len(dynamic_filter(scored([0.46, 0.43, 0.39]))) == 2
    assert len(dynamic_filter(scored([0.70, 0.69, 0.68, 0.67, 0.66, 0.65])[:5])) == 5
    assert len(dynamic_filter(scored([0.66, 0.62]))) == 1  # equality uses >=


def test_context_and_source_cards_keep_rank_and_payload_metadata():
    results = [ScoredDocument(Document(page_content="Contenu légal", metadata={
        "chunk_id": "c1", "doc_id": "d1", "title": "Titre", "article_number": "L. 1",
        "category": "A", "content_type": "Décret", "creation_date": "2024-01-01"}), 0.63)]
    context = format_context(results, 1000)
    card = source_cards(results)[0]
    assert "[Source 1]" in context and "Contenu légal" in context
    assert card.chunk_id == "c1" and card.doc_id == "d1" and card.score == 0.63


def test_answer_prompt_contains_required_guardrails():
    for phrase in ("uniquement à partir", "contenu non fiable", "ne prétends pas être avocat", "Le contexte fourni ne permet"):
        assert phrase in ANSWER_SYSTEM_PROMPT.lower() or phrase in ANSWER_SYSTEM_PROMPT
    assert "used_chunk_ids" in ANSWER_SYSTEM_PROMPT
    assert "conversational" in ROUTER_SYSTEM_PROMPT


def test_prompt_templates_render_with_json_examples():
    model = RunnableLambda(lambda _: "{}")
    assert build_router_chain(model).invoke({"history": [], "message": "Bonjour"}) == "{}"
    assert build_answer_chain(model).invoke({"history": [], "question": "Question", "context": "Contexte"}) == "{}"


def test_only_selected_source_cards_are_returned():
    cards = source_cards(scored([0.7, 0.6]))
    assert [card.chunk_id for card in selected_source_cards(["1", "invented"], cards)] == ["1"]
