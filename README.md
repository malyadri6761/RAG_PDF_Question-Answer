# Multimodal Document & Image Assistant

A RAG (Retrieval-Augmented Generation) assistant that answers questions over **PDFs and images**,
with **source citations** (file name and page number), served through a **FastAPI** backend.

> 🚧 Work in progress. See the [Roadmap](#roadmap).

## Features (planned)
- Upload PDFs and images (PNG, JPG, WEBP, ...)
- Extract text from PDF pages, scanned pages (OCR), and embedded images
- Describe charts and diagrams with a vision LLM so they can be searched
- Semantic search over all content using embeddings
- Answers grounded only in your documents, with `[n]` citations to file and page
- Says "not found" instead of guessing when the documents don't contain the answer

## How it works
```
INGEST:  file → loader (PDF text / OCR / image caption) → chunks (with file + page) → embeddings → vector index
QUERY:   question → embedding → top-k similar chunks → LLM answers using only those → answer + citations
```

## Tech stack
| Part | Tool |
|---|---|
| API | FastAPI, Uvicorn |
| PDF parsing | PyMuPDF |
| OCR | Tesseract (pytesseract) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | NumPy cosine similarity |
| LLM | Claude / OpenAI / Gemini (configurable) |
| Packaging | uv |

## Project structure
```
app/
├── ingestion/   file → chunks (models, chunker, image_loader, pdf_loader, pipeline)
├── retrieval/   embeddings + vector store
├── llm/         LLM client + image captioner
├── rag/         question → cited answer
├── core/        shared services
├── api/         FastAPI routes
└── main.py
```

## Roadmap
- [ ] Phase 0: Project setup (uv, folder structure)
- [ ] Phase 1: Ingestion (PDF text, OCR, embedded images, chunking)
- [ ] Phase 2: Embeddings and vector store
- [ ] Phase 3: LLM client and image captioning
- [ ] Phase 4: Question answering with citations
- [ ] Phase 5: FastAPI endpoints (`/ingest`, `/query`, `/documents`)
- [ ] Phase 6: Tests and evaluation

## Setup
Requires [uv](https://docs.astral.sh/uv/) and [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).

```bash
git clone https://github.com/<your-username>/multimodal-rag.git
cd multimodal-rag
uv sync
```

Create a `.env` file:
```
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-5-5
ANTHROPIC_API_KEY=your-key
```

Run (once Phase 5 is done):
```bash
uv run uvicorn app.main:app --reload
```
Then open http://127.0.0.1:8000/docs
