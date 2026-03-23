# PROGRESS.md — rag-llama

Running log of all additions and changes to the project. Most recent entries at the top.

---

## 2026-03-22 — Initial Setup

### Added
- `CLAUDE.md` — project context and conventions for Claude Code
- `PROGRESS.md` — this file; running log of project changes
- `.env` — environment config (OLLAMA_BASE_URL, CHROMA_PATH, DATA_PATH, PORT)
- `requirements.txt` — all Python dependencies
- `ingest.py` — Excel → ChromaDB ingestion pipeline
  - Recursively finds all `.xlsx` files under `data/`
  - Reads all sheets per file via `pandas.read_excel(sheet_name=None)`
  - Converts each row to a LlamaIndex `Document` with source/sheet/row metadata
  - Embeds using `nomic-embed-text` via Ollama
  - Persists vectors to ChromaDB at `./chroma_db`
  - Skips re-ingestion if `chroma_db/` already exists
- `server.py` — FastAPI query server
  - Loads ChromaDB on startup
  - `GET /health` → `{"status": "ok"}`
  - `POST /ask` → `{"answer": "..."}` using `llama3.2` via Ollama
  - CORS open for LAN phone access
  - Prints local and LAN IP on startup
- `iiot/` — Python virtual environment with all dependencies installed

### Data ingested
- 119 shift report `.xlsx` files from `data/2026_Spring_ME597/KRPM_shift_report_Apr-June2025/`
- `data/2026_Spring_ME597/MTConnect_data/krpm.operation_20250901.xlsx`
- Large CSVs excluded (4+ GB)

### Stack
- LlamaIndex 0.14.18 + ChromaDB 1.5.5 + FastAPI 0.135.1 + Ollama

---

## 2026-03-23 — GitHub Files

### Added
- `README.md` — project overview, setup instructions, usage, and API reference for GitHub
- `.gitignore` — excludes `iiot/`, `chroma_db/`, `.env`, `__pycache__`

---

## 2026-03-23 — Conversation Memory

### Changed
- `server.py`: replaced `query_engine` with `chat_engine` using `condense_plus_context` mode
  - Follow-up questions ("Got any more info on that?") are now rewritten using chat history before querying ChromaDB
  - Uses `index.as_chat_engine(chat_mode="condense_plus_context", ...)`
- Added `POST /reset` endpoint to clear conversation history
- Added "New Chat" button to the UI header — calls `/reset` and clears visible messages

---

## 2026-03-23 — Chat UI

### Added
- `GET /` in `server.py` — serves a self-contained chat UI (no new files or dependencies)
  - Chat bubble layout: user messages right (blue), bot answers left (white)
  - "Thinking..." indicator while waiting for a response
  - Send on button click or Enter key (Shift+Enter for newline)
  - Auto-resizing textarea, mobile-friendly full-viewport layout
  - Error bubble shown if the server is unreachable or returns an error

---
<!-- Add new entries above this line, newest first -->
