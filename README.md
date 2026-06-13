# High-Performance PDF RAG Engine

A lightweight, production-ready Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **ChromaDB**, and **Groq**. This project is engineered with clean separation of concerns, constructor dependency injection, and optimized local embedding generation.

---

## 🛠️ Tech Stack & Key Choices

- **Web Framework:** FastAPI (Asynchronous request handling, Pydantic validation, OpenAPI/Swagger docs generation)
- **Vector Database:** ChromaDB (Persisted local storage, HNSW indexing for fast similarity searches)
- **Embedding Model:** SentenceTransformers (`all-MiniLM-L6-v2`) — a highly optimized 384-dimensional CPU-friendly local model (80MB memory footprint)
- **LLM Provider:** Groq Cloud SDK (LLaMA 3.1 8B via ultra-low latency LPU inference)
- **PDF Processing:** PyPDF2 (Stateless parsing & custom sliding-window text chunker)

---

## 🏗️ Architecture & Software Design

The project is structured with an emphasis on **clean architecture** and **dependency injection (DI)**. This makes the system modular, testable, and vendor-agnostic.

```
app/
├── config.py         # Type-safe settings using native dataclasses (loads and validates .env)
├── llm.py            # Isolated LLM client interface (vendor-agnostic wrapper)
├── embeddings.py     # Independent text embedding service
├── vector_store.py   # ChromaDB wrapper; accepts EmbeddingModel via constructor injection
├── pdf_reader.py     # Stateless PDF text extraction & sliding-window chunking
├── routes.py         # FastAPI endpoints defined via a configurable router factory
└── main.py           # Single entrypoint handling explicit object instantiation and wiring
```

### Key Software Engineering Design Patterns:
1. **Dependency Injection:** The `VectorStore` class requires an `EmbeddingModel` instance in its constructor. The router factory `create_router` explicitly accepts its `LLMClient` and `VectorStore` dependencies. This avoids global singletons and makes unit testing with mock objects trivial.
2. **Configuration Validation:** Configuration is loaded into a typed `Settings` dataclass. Missing required environment variables (e.g., `GROQ_API_KEY`) cause the app to fail-fast at startup rather than throwing runtime exceptions mid-request.
3. **Decoupled Embedding Engine:** The embedding logic is isolated from the vector database. If you want to switch from local SentenceTransformers to OpenAI/Cohere API embeddings, only `app/embeddings.py` needs to be modified.

---

## 🚀 API Endpoints

Once the application is running, FastAPI automatically generates interactive OpenAPI documentation at `/docs`.

### 1. Ingestion Pipeline
- **Endpoint:** `POST /api/upload`
- **Payload:** `multipart/form-data` containing a PDF file.
- **Workflow:** Extracts text → Chunks text using sliding-window algorithm (500 chars window, 50 chars overlap) → Computes dense vectors → Stores in ChromaDB with metadata.

### 2. Retrieval & Generation Pipeline
- **Endpoint:** `POST /api/ask`
- **Payload:** `{"question": "string", "collection_name": "string"}`
- **Workflow:** Embeds query → Searches ChromaDB using Cosine similarity → Retrieves top $K$ relevant chunks → Inject chunks into system prompt (Context-grounding) → Generates response via LLaMA 3.1.

---

## 💻 Quick Start

### 1. Prerequisites
- Python 3.11 or higher
- Git

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/rag-engine.git
cd rag-engine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.1-8b-instant
LLM_TEMPERATURE=0.2
VECTOR_STORE_PATH=./chroma_data
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 4. Running the Application
Start the FastAPI server:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000/docs` to test endpoints interactively.
