"""
test_llm.py — Quick Sanity Test

Run this to verify your Groq API key works and the LLM responds.
    python test_llm.py
"""

from app.config import settings
from app.llm import LLMClient

if __name__ == "__main__":
    print(f"Model:  {settings.llm_model}")
    print(f"Temp:   {settings.llm_temperature}")
    print("-" * 40)

    llm = LLMClient()

    response = llm.generate(
        system_prompt="You are a helpful assistant. Answer in 2 lines max.",
        user_message="What is RAG in the context of AI?",
        temperature=0.2,
    )

    print("Response:")
    print(response)
