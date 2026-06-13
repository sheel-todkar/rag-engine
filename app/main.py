"""
main.py — Application Entrypoint

WHY THIS EXISTS:
    This is the ONLY file that knows about all the pieces.
    It creates every component in the right order and wires them together.
    Every other file is independent — it doesn't know who created it.

INTERVIEW POINT:
    "I initialize everything explicitly in main.py so you can read the
    startup sequence top-to-bottom. No hidden singletons, no import-time
    side effects (except config loading). This makes debugging startup
    issues trivial."

INITIALIZATION ORDER:
    1. Settings     (config.py)   — loaded at import time, validates .env
    2. EmbeddingModel              — heaviest step, loads ML model into RAM
    3. VectorStore                 — lightweight, just creates ChromaDB client
    4. LLMClient                   — lightweight, stores API key
    5. Router                      — wires routes with dependencies
    6. FastAPI app                 — mounts the router
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore
from app.llm import LLMClient
from app.routes import create_router

# --- Initialize Components ---
# Order matters: VectorStore needs EmbeddingModel, routes need everything.

embedding_model = EmbeddingModel()
vector_store = VectorStore(embedding_model)
llm_client = LLMClient()

# --- Create FastAPI App ---

app = FastAPI(
    title="RAG Engine API",
    description="Upload PDFs and ask questions — answers are grounded in your documents.",
    version="1.0.0",
)

# --- Mount Static Files ---
# Serve frontend assets (HTML, CSS, JS) from the static folder
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# --- Wire Routes ---

router = create_router(llm=llm_client, vector_store=vector_store)
app.include_router(router)


@app.get("/")
def root():
    """Serve the single-page RAG web interface."""
    return FileResponse(os.path.join("app", "static", "index.html"))


# --- Direct Run Support ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
