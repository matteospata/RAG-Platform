from __future__ import annotations

from abc import ABC, abstractmethod


SYSTEM_PROMPT = """You are a reliable RAG assistant.
Answer exclusively using the provided context.
If the context does not contain the answer, clearly state that there is not enough information.
Do not invent sources, facts, or citations. Cite sources using the format [S1], [S2].
Answer in English with a clear and concise summary.
"""


class ChatModel(ABC):
    name: str

    @abstractmethod
    def answer(self, question: str, context: str) -> str:
        raise NotImplementedError


class ExtractiveChatModel(ChatModel):
    """Free fallback useful for smoke tests and demos without an API key."""

    def __init__(self) -> None:
        self.name = "extractive-local"

    def answer(self, question: str, context: str) -> str:
        if not context.strip():
            return "I could not find enough information in the knowledge base."
        sections = [section.strip() for section in context.split("\n\n") if section.strip()]
        question_terms = {term.lower() for term in question.split() if len(term) > 3}
        ranked = sorted(
            sections,
            key=lambda section: sum(term in section.lower() for term in question_terms),
            reverse=True,
        )
        return "\n\n".join(ranked[:3])


class OpenAIChatModel(ChatModel):
    def __init__(self, api_key: str, model_name: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'llm' extra to use OpenAI") from exc
        self.client = OpenAI(api_key=api_key)
        self.name = model_name

    def answer(self, question: str, context: str) -> str:
        response = self.client.responses.create(
            model=self.name,
            instructions=SYSTEM_PROMPT,
            input=f"CONTESTO:\n{context}\n\nDOMANDA:\n{question}",
        )
        return response.output_text.strip()


def build_chat_model(provider: str, model_name: str, api_key: str | None) -> ChatModel:
    if provider == "extractive":
        return ExtractiveChatModel()
    if provider == "openai":
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the openai LLM provider")
        return OpenAIChatModel(api_key, model_name)
    raise ValueError(f"Unknown LLM provider: {provider}")
