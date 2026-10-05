"""Small LCEL builders; models are injected at application startup."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from app.rag.prompts import ANSWER_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT


def build_router_chain(model: ChatOllama):
    """Build the first LLM call: conversation routing plus legal query rewriting."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_SYSTEM_PROMPT),
            MessagesPlaceholder("history", optional=True),
            ("human", "Dernier message : {message}"),
        ]
    )
    return prompt | model | StrOutputParser()


def build_answer_chain(model: ChatOllama):
    """Return the grounded answer chain using already formatted legal context."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            MessagesPlaceholder("history", optional=True),
            (
                "human",
                "Question autonome : {question}\n\nContexte juridique :\n{context}",
            ),
        ]
    )
    return prompt | model | StrOutputParser()
