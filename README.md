# rag-llama

A local RAG (Retrieval-Augmented Generation) application for querying IIoT/manufacturing data — shift reports, operation logs, and acoustic sensor recordings — via a conversational chat interface. Runs entirely on your machine — no cloud, no API keys.

## Stack

| Layer | Technology |
|-------|-----------|
| RAG pipeline | [LlamaIndex](https://www.llamaindex.ai/) |
| LLM & embeddings | [Ollama](https://ollama.com/) — `llama3.2` + `nomic-embed-text` |
| Vector store | [ChromaDB](https://www.trychroma.com/) (persisted locally) |
| API server | [FastAPI](https://fastapi.tiangolo.com/) |
| Tabular data | Excel `.xlsx` files (shift reports, operation logs) |
| Audio data | WAV sensor recordings — features extracted and indexed as text |
| Audio processing | `librosa`, `scipy` — RMS, kurtosis, crest factor, FFT-based features |

## Features

- Chat UI accessible from any browser on the local network, including mobile
- Conversation memory — follow-up questions understand context from previous messages
- "New Chat" button to reset the conversation
- Recursively ingests all `.xlsx` files, reading every sheet automatically
- Ingests WAV audio files — extracts acoustic features and fault flags, stores as searchable text chunks
- Correlates shift report data with sensor audio events by timestamp
- Skips re-ingestion if the vector store already exists

## Project Structure

```
rag-llama/
├── data/                  ← place your .xlsx and .wav files here (subdirectories supported)
├── chroma_db/             ← auto-generated after running ingest.py
├── iiot/                  ← Python virtual environment
├── ingest.py              ← Excel + WAV → ChromaDB ingestion pipeline
├── server.py              ← FastAPI server + chat UI
├── audio_processing.py    ← WAV feature extraction pipeline (used by ingest.py)
├── requirements.txt
├── .env
├── CLAUDE.md              ← context file for Claude Code
└── PROGRESS.md            ← running changelog
```

## Prerequisites

**1. Install Ollama**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**2. Pull the required models**
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Setup

**1. Clone and enter the repo**
```bash
git clone <repo-url>
cd rag-llama
```

**2. Create the virtual environment and install dependencies**
```bash
python3 -m venv iiot
source iiot/bin/activate
pip install -r requirements.txt
```

**3. Add your data**

Place your `.xlsx` files anywhere under the `data/` folder. Subdirectories are supported.

**4. Configure environment (optional)**

Defaults in `.env` work out of the box:
```
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PATH=./chroma_db
DATA_PATH=./data
PORT=8000
```

## Usage

**Step 1 — Ingest your data** (first time, or after adding new files)
```bash
source iiot/bin/activate
python ingest.py
```

This reads all `.xlsx` files (each row becomes a document) and all `.wav` audio files (acoustic features extracted and serialized as text). Everything is embedded using `nomic-embed-text` and stored in ChromaDB. Run time depends on data volume. Re-running is a no-op if `chroma_db/` already exists — delete that folder to force re-ingestion.

**Step 2 — Start the server**
```bash
python server.py
```

The server prints your local and LAN addresses on startup:
```
=== Server ready ===
  Local:  http://localhost:8000
  Phone:  http://192.168.x.x:8000
  Docs:   http://localhost:8000/docs
```

**Step 3 — Open the chat UI**

Navigate to `http://localhost:8000` in any browser. On your phone, use the LAN address printed above (must be on the same Wi-Fi network).

## API

| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| `GET` | `/` | — | Chat UI (HTML) |
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/ask` | `{"question": "..."}` | `{"answer": "..."}` |
| `POST` | `/reset` | — | Clears conversation history |

## Audio Processing

WAV files are processed by `audio_processing.py` before ingestion. The pipeline has three stages:

1. **Load & normalize** — reads the WAV file at its native sample rate, normalizes amplitude to `[-1, 1]`
2. **Feature extraction** — computes:
   - `RMS` — overall signal energy
   - `Kurtosis` — statistical spikiness (indicator of impacts/faults)
   - `Crest factor` — peak-to-RMS ratio
   - `Dominant frequency` — frequency band carrying the most energy (Hz)
   - `Low-band energy ratio` — fraction of energy in the 80–500 Hz band
   - `Fault flag` — `True` when kurtosis > 4 AND crest factor > 5
3. **Serialization** — features are written as a plain-text chunk with the sensor filename and timestamp, ready for vector embedding

Filename format expected: `YYYYMMDD_HHMMSS_<suffix>_mb4000_<sensor>.wav`

## Notes

- Only `.xlsx` and `.wav` files are ingested — large CSVs and other formats are ignored
- The `iiot/` virtual environment folder should be added to `.gitignore`
- The `chroma_db/` folder contains your vector index and can be regenerated at any time by deleting it and re-running `ingest.py`
