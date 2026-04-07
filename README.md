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

1. Clone the repository:

```bash
git clone https://github.com/Reboot2004/Multi-modal-RAG-Langraph.git
cd Multi-modal-RAG-Langraph
```

2. Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\venv\Scripts\Activate.ps1
```

Windows CMD:

```bat
python -m venv venv
venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create your environment file:

```bash
cp .env.example .env
```

If `cp` is not available on Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

5. Add API keys to `.env`:

- `GROQ_API_KEY` (required for Groq)
- `OPENROUTER_API_KEY` (optional)

API key links:

- Groq Console: https://console.groq.com/keys
- Groq Signup: https://console.groq.com/
- OpenRouter Keys: https://openrouter.ai/keys
- OpenRouter Signup: https://openrouter.ai/

Example `.env`:

```env
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

## Run

Start Streamlit:

```bash
streamlit run app.py
```

Alternative for Windows:

```bash
run_app.bat
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

## First Run Checklist

1. Virtual environment is active.
2. Dependencies installed successfully.
3. `.env` exists and contains at least `GROQ_API_KEY`.
4. `streamlit run app.py` starts without import errors.

## Notes

- `.env`, `venv/`, `data/`, caches, and local artifacts are excluded from git via `.gitignore`.
- Keep secrets only in environment files/variables, never in source code.
