# Multi-modal-RAG-Langraph

Multilingual, multimodal Retrieval-Augmented Generation (RAG) system built with Streamlit, LangGraph, FAISS, and Groq/OpenRouter LLM backends.

Repository: https://github.com/Reboot2004/Multi-modal-RAG-Langraph

## Features

- Multimodal ingestion: PDF, images, text/code files, DOCX, CSV, Excel, HTML, JSON, PPT/PPTX, audio, and video.
- Hybrid retrieval: semantic FAISS + lexical + HyPE fusion.
- Query orchestration with LangGraph.
- Self-RAG quality gates: relevance, faithfulness, usefulness, and confidence scoring.
- Tiered enhancements:
  - Tier 1: adaptive retrieval and LLM reranking.
  - Tier 2: fallback retrieval, semantic cache, citation tracking.
  - Tier 3: iterative agentic answer refinement.
- Multilingual response handling for Indic and English workflows.

## Tech Stack

- UI: Streamlit
- Orchestration: LangGraph
- Embeddings: sentence-transformers (`BAAI/bge-m3`)
- Vector DB: FAISS
- Reranker: cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`)
- LLM providers:
  - Groq (`llama-3.1-8b-instant` default)
  - OpenRouter (`nvidia/nemotron-nano-9b-v2:free` optional)
- OCR: PaddleOCR
- ASR: faster-whisper

## Project Structure

- `app.py`: Streamlit entrypoint
- `config/`: runtime settings and feature toggles
- `ingestion/`: document/audio/video parsing pipelines
- `processing/`: cleaning, chunking, language detection
- `embeddings/`: embedding and vector store components
- `retrieval/`: retriever and reranking logic
- `orchestration/`: LangGraph flows, Self-RAG gates, Tier 3 refinement
- `llm/`: provider clients and prompt builder
- `utils/`: helper modules (cache, conversation memory, citations)

## Setup

1. Create and activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables (copy from `.env.example`):

- `GROQ_API_KEY` (required for Groq)
- `OPENROUTER_API_KEY` (optional)

## Run

```bash
streamlit run app.py
```

or use:

```bash
run_app.bat
```

## Notes

- `.env`, `venv/`, `data/`, caches, and local artifacts are excluded from git via `.gitignore`.
- Keep secrets only in environment files/variables, never in source code.
