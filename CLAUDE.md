# CLAUDE.md — rag-llama

## Project Overview
RAG (Retrieval-Augmented Generation) application over IIoT/manufacturing data.
Queries are served via FastAPI so they can be called from a phone on the local network.

## Stack
- **LlamaIndex** — RAG pipeline (indexing + querying), `chat_mode="condense_plus_context"`, top-k=5
- **Ollama** — local LLM (`llama3.2`) and embedding model (`nomic-embed-text`)
- **ChromaDB** — persistent vector store at `./chroma_db`
- **FastAPI** — REST API server on `0.0.0.0:8000` with built-in chat UI at `/`
- **Pandas + openpyxl** — Excel ingestion
- **librosa + scipy** — WAV audio feature extraction (audio_processing.py)

## Virtual Environment
All dependencies live in the `iiot/` venv.
```bash
source iiot/bin/activate
```

## Key Files
| File | Purpose |
|------|---------|
| `ingest.py` | Reads all `.xlsx` and `.wav` files from `data/`, embeds rows/audio chunks, stores in ChromaDB |
| `server.py` | FastAPI server — loads ChromaDB, exposes `/ask`, `/health`, `/reset`, and the built-in chat UI at `/` |
| `audio_processing.py` | WAV feature extraction pipeline — 3 stages: load/normalize → extract RMS/kurtosis/crest factor/dominant freq/low-band energy/fault flag → serialize to RAG text chunk |
| `.env` | Runtime config (OLLAMA_BASE_URL, CHROMA_PATH, DATA_PATH, PORT) |
| `requirements.txt` | Python dependencies |
| `PROGRESS.md` | Running log of changes and additions |

## Data
- Source: `data/2026_Spring_ME597/`
  - `KRPM_shift_report_Apr-June2025/` — 119 shift report `.xlsx` files
  - `MTConnect_data/krpm.operation_20250901.xlsx` — operation log
- Large CSVs in `MTConnect_data/` are **excluded** (4+ GB, not ingested)
- `chroma_db/` is the persisted vector DB — delete it to force re-ingestion

## Running the App
```bash
source iiot/bin/activate

# First time (or after new data): ingest
python ingest.py

# Start the server
python server.py
```

## API Endpoints
| Method | Path | Body | Response |
|--------|------|------|----------|
| GET | `/` | — | Chat UI (HTML, mobile-friendly) |
| GET | `/health` | — | `{"status": "ok"}` |
| POST | `/ask` | `{"question": "..."}` | `{"answer": "..."}` |
| POST | `/reset` | — | `{"status": "conversation cleared"}` |

## Environment Variables (`.env`)
```
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PATH=./chroma_db
DATA_PATH=./data
PORT=8000
```

## Prerequisites
Ollama must be running with both models pulled:
```bash
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Coding Conventions
- Keep ingestion and serving as separate scripts (`ingest.py` / `server.py`)
- All config via `.env` — no hardcoded paths or URLs
- Print progress during ingestion so the user can follow along
- CORS is open (`*`) — this is intentional for LAN phone access
