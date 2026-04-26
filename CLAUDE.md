# CLAUDE.md — rag-llama

## Project Overview
RAG (Retrieval-Augmented Generation) application over IIoT/manufacturing data.
Queries are served via FastAPI so they can be called from a phone on the local network.

## Stack
- **LlamaIndex** — RAG pipeline (indexing + querying), `chat_mode="condense_plus_context"`, top-k=5
- **Ollama** — local LLM (`llama3.2`) and embedding model (`nomic-embed-text`)
- **ChromaDB** — persistent vector store at `./chroma_db`, collection name `iiot_rag`
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
| `ingest.py` | Loads `.md` KB, `.xlsx`, sampled `.csv`, and `.wav` files from `data/`, embeds into ChromaDB |
| `server.py` | FastAPI server — loads ChromaDB (`iiot_rag`), exposes `/ask`, `/health`, `/reset`, chat UI at `/` |
| `audio_processing.py` | Full 10-feature WAV pipeline — load/normalize → extract features → interpret → RAG chunk |
| `mb4000_troubleshooting.md` | Diagnostic knowledge base for MB4000 CNC machine — ingested automatically |
| `.env` | Runtime config (OLLAMA_BASE_URL, CHROMA_PATH, DATA_PATH, KB_PATH, CSV_SAMPLE_EVERY, PORT) |
| `requirements.txt` | Python dependencies |
| `PROGRESS.md` | Running log of changes and additions |

## Data
- `data/Sounds/250428/` — MB6000 WAV recordings for Apr 28–29, 2025 (902 files)
- `data/Sounds/250527/` — MB6000 WAV recordings for May 27–28, 2025 (1261 files)
- `data/pivoted_metrics_250428-0513_filtered (1).csv` — MTConnect data Apr 28–May 13 (sampled)
- `data/pivoted_metrics_250527-0614_filtered (1).csv` — MTConnect data May 27–Jun 14 (sampled)
- `data/Job_enriched_ar.xlsx` and `data/file_with_utc_shift_times_Ar.xlsx` — enriched job/shift data
- `data/2026_Spring_ME597/` — original shift reports and MTConnect xlsx (legacy)
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
KB_PATH=./
CSV_SAMPLE_EVERY=60
PORT=8000
```

## Prerequisites
Ollama must be running with both models pulled:
```bash
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Audio Pipeline (audio_processing.py)
10 features extracted per WAV file:
| Feature | What it detects |
|---------|----------------|
| RMS | Overall energy / loudness |
| Kurtosis | Impulsive spikes — bearing/gear faults |
| Crest Factor | Peak-to-average ratio — impact events |
| Dominant Frequency | Main frequency in Hz |
| Spectral Centroid | Centre of mass of spectrum — rises with wear/friction |
| Spectral Bandwidth | Spread of energy — wide = broadband fault |
| Low-band Energy Ratio | Fraction of energy in 80–500 Hz |
| Harmonic Ratio | Tonal vs noisy — drops with looseness |
| Zero-Crossing Rate | High-frequency chattering |
| MFCC Delta | Rate of spectral change — unstable operation |

Severity levels: `normal` | `warning` | `fault`
Fault flag: kurtosis > 4.0 AND crest factor > 5.0

## Ingest Pipeline (ingest.py)
Processes four data types in order:
1. `.md` / `.txt` files from `KB_PATH` — knowledge base (split on `---` separators)
2. `.xlsx` files from `DATA_PATH` — MTConnect rows as natural-language documents with fault flags
3. `.csv` files from `DATA_PATH` — sampled every `CSV_SAMPLE_EVERY` rows (default 60)
4. `.wav` files from `DATA_PATH` — processed through `audio_processing.py`

## Coding Conventions
- Keep ingestion and serving as separate scripts (`ingest.py` / `server.py`)
- All config via `.env` — no hardcoded paths or URLs
- Print progress during ingestion so the user can follow along
- CORS is open (`*`) — this is intentional for LAN phone access
- ChromaDB collection name: `iiot_rag`
