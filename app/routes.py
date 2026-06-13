"""
routes.py — API Endpoints

WHY THIS EXISTS:
    Separates HTTP handling from business logic. The routes only do:
    1. Receive the request
    2. Call the right service
    3. Return the response
    No business logic lives here.

INTERVIEW POINT:
    "I use FastAPI because it auto-generates OpenAPI docs at /docs,
    has native async support, and validates request/response bodies
    with Pydantic models — less boilerplate, fewer bugs."
"""

import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.llm import LLMClient
from app.vector_store import VectorStore
from app import pdf_reader


# --- Request / Response Models ---

class AskRequest(BaseModel):
    """What the client sends to /api/ask"""
    question: str
    collection_name: str = "documents"


class AskResponse(BaseModel):
    """What /api/ask returns"""
    answer: str
    sources: list[str]


# --- Route Factory ---
# We use a factory function so routes receive their dependencies explicitly.
# This avoids global variables and makes testing easy (pass in mocks).

def create_router(llm: LLMClient, vector_store: VectorStore) -> APIRouter:
    """
    Create an APIRouter with all RAG endpoints.

    Args:
        llm:          LLM client for generating answers.
        vector_store: Vector store for document storage/retrieval.
    """
    router = APIRouter(prefix="/api", tags=["RAG"])

    @router.get("/health")
    def health_check():
        """Simple liveness check — returns 200 if the server is running."""
        return {"status": "healthy"}

    @router.post("/upload")
    async def upload_pdf(
        file: UploadFile = File(...),
        collection_name: str = "documents",
    ):
        """
        Upload a PDF → extract text → chunk it → embed & store in vector DB.

        This is the INGESTION pipeline:
        PDF file → raw text → chunks → embeddings → ChromaDB
        """
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        try:
            # 1. Save the uploaded file to disk
            upload_dir = "./uploads"
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, file.filename)

            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # 2. Extract text from PDF
            text = pdf_reader.extract_text(file_path)
            if not text.strip():
                raise HTTPException(status_code=400, detail="PDF contains no extractable text")

            # 3. Split into overlapping chunks
            chunks = pdf_reader.chunk_text(text)

            # 4. Embed chunks and store in vector database
            vector_store.add_documents(collection_name, chunks)

            return {
                "message": "PDF processed successfully",
                "filename": file.filename,
                "chunks_created": len(chunks),
            }
        except HTTPException:
            raise  # Re-raise HTTP exceptions as-is
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/ask", response_model=AskResponse)
    async def ask_question(request: AskRequest):
        """
        Ask a question → search vector DB → LLM generates answer from context.

        This is the RETRIEVAL + GENERATION pipeline:
        Question → embed → search ChromaDB → get top chunks → LLM answers
        """
        try:
            # 1. Retrieve relevant document chunks
            sources = vector_store.search(request.collection_name, request.question)

            # 2. Combine chunks into a single context string
            context = "\n\n".join(sources)

            # 3. Send question + context to LLM
            answer = llm.generate_with_context(request.question, context)

            return AskResponse(answer=answer, sources=sources)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
