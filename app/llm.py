"""
llm.py — LLM Client Wrapper

WHY THIS EXISTS:
    Isolates the LLM vendor (Groq) behind a clean interface.
    If we switch to OpenAI or Anthropic tomorrow, only THIS file changes.
    The rest of the app just calls generate() and doesn't care who's behind it.

INTERVIEW POINT:
    "I separated the raw LLM call from the RAG prompt so each can be
    tested independently. The RAG prompt uses a strict instruction to
    prevent hallucination — the LLM must answer ONLY from provided context."
"""

from groq import Groq
from app.config import settings


class LLMClient:
    """Wrapper around the Groq inference API."""

    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.llm_model

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
    ) -> str:
        """
        Send a prompt to the LLM and return the response text.

        Args:
            system_prompt: Instructions for the LLM's behavior.
            user_message:  The user's actual question or input.
            temperature:   Creativity dial (0=deterministic, 1=creative).
                           Defaults to the value from .env.
        """
        if temperature is None:
            temperature = settings.llm_temperature

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def generate_with_context(self, question: str, context: str) -> str:
        """
        RAG-specific generation: answer a question using ONLY the provided context.

        This is where RAG happens — we inject retrieved document chunks
        into the system prompt so the LLM has real data to work with.
        """
        system_prompt = (
            "You are a helpful assistant. Answer the user's question "
            "based ONLY on the context provided below. "
            "If the context does not contain enough information, say so clearly.\n\n"
            f"CONTEXT:\n{context}"
        )
        # Low temperature for factual, grounded answers
        return self.generate(system_prompt, question, temperature=0.2)
