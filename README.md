# rag-llama

A local RAG (Retrieval-Augmented Generation) application for querying IIoT/manufacturing data — shift reports, MTConnect operation logs, CSV time-series, and MB6000 acoustic sensor recordings — via a conversational chat interface. Runs entirely on your machine — no cloud, no API keys.

## Stack

| Layer | Technology |
|-------|-----------|
| RAG pipeline | [LlamaIndex](https://www.llamaindex.ai/) |
| LLM & embeddings | [Ollama](https://ollama.com/) — `llama3.2` + `nomic-embed-text` |
| Vector store | [ChromaDB](https://www.trychroma.com/) (persisted locally, collection `iiot_rag`) |
| API server | [FastAPI](https://fastapi.tiangolo.com/) |
| Tabular data | Excel `.xlsx` + sampled `.csv` MTConnect files |
| Audio data | MB6000 WAV sensor recordings — 10 acoustic features extracted and indexed as text |
| Audio processing | `librosa`, `scipy` |

## Features

- Chat UI accessible from any browser on the local network, including mobile
- Conversation memory — follow-up questions understand context from previous messages
- Ingests four data types: `.md` knowledge base, `.xlsx`, sampled `.csv`, and `.wav` audio
- MB4000 diagnostic knowledge base (`mb4000_troubleshooting.md`) always available as context
- 10-feature acoustic analysis per WAV file with severity classification (`normal` / `warning` / `fault`)
- MTConnect fault flags (HIGH-SPINDLE-LOAD, SPINDLE-OVERLOAD, ESTOP-TRIGGERED) always preserved regardless of sampling interval
- Per-CSV program frequency summary injected at ingest time so "most common CNC program" queries resolve correctly
- Fine-grained metadata types per Excel sheet (`part_details`, `machine_summary`, `shift_summary`, etc.) enabling precise filtered retrieval
- Intelligent query routing in server: part/job queries → `parts_engine`, employee queries → `employee_engine`, audio queries → `audio_engine`
- Custom QA prompt forces explicit value citation (part numbers, job orders, program names) instead of pronoun-only answers
- CSV sampling — configurable row interval keeps large time-series manageable
- `benchmark.py` — 16-question accuracy benchmark grounded in real data, scores by keyword match, saves JSON report

## Project Structure

```
rag-llama/
├── data/                          ← place your data files here (subdirectories supported)
│   ├── Sounds/250428/             ← MB6000 WAV recordings Apr 28–29
│   ├── Sounds/250527/             ← MB6000 WAV recordings May 27–28
│   ├── *.csv                      ← MTConnect filtered time-series
│   └── *.xlsx                     ← enriched job/shift data
├── chroma_db/                     ← auto-generated after running ingest.py
├── iiot/                          ← Python virtual environment
├── ingest.py                      ← KB + Excel + CSV + WAV → ChromaDB pipeline
├── server.py                      ← FastAPI server + chat UI + query routing
├── benchmark.py                   ← 16-question RAG accuracy benchmark
├── audio_processing.py            ← WAV 10-feature extraction pipeline
├── mb4000_troubleshooting.md      ← MB4000 diagnostic knowledge base
├── requirements.txt
├── .env
└── CLAUDE.md                      ← context file for Claude Code
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

Place `.xlsx`, `.csv`, and `.wav` files anywhere under the `data/` folder. Subdirectories are supported.

**4. Configure environment**

Defaults in `.env` work out of the box:
```
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PATH=./chroma_db
DATA_PATH=./data
KB_PATH=./
CSV_SAMPLE_EVERY=60
PORT=8000
```

`KB_PATH` — folder containing `.md` knowledge base files (default: project root).
`CSV_SAMPLE_EVERY` — ingest 1 row per N rows from CSV files (default: 60 = 1-minute intervals for 1s data).

## Usage

**Step 1 — Start Ollama**
```bash
ollama serve
```

**Step 2 — Ingest your data** (first time, or after adding new files)
```bash
source iiot/bin/activate
python ingest.py
```

Expected output:
```
── Loading Knowledge Base ─────────────────────────────────
  [kb]  mb4000_troubleshooting.md — 13 sections

[scan]  2 Excel files, 2 CSV files, 2163 WAV files

── Ingesting Excel files ──────────────────────────────────
── Ingesting CSV files (every 60 rows) ────────────────────
── Ingesting WAV files ────────────────────────────────────
  [NORMAL ]  20250428_100005.513405Z_mb6000_sensor1.wav
  [WARNING]  20250428_121304.090144Z_mb6000_sensor1.wav
             -> Kurtosis (6.2) exceeds the fault threshold...

[done]  X documents embedded into ./chroma_db
```

To force a full re-ingest, delete `chroma_db/` first:
```bash
rm -rf chroma_db/
python ingest.py
```

**Step 2b — Run the accuracy benchmark (optional)**
```bash
python benchmark.py --verbose
```

Prints a per-category score report and saves a timestamped JSON result. No server needed.

**Step 3 — Start the server**
```bash
python server.py
```

The server prints your local and LAN addresses on startup:
```
=== Server ready ===
  Local:  http://localhost:8000
  Phone:  http://192.168.x.x:8000
```

**Step 4 — Open the chat UI**

Navigate to `http://localhost:8000` in any browser. On your phone, use the LAN address (must be on the same Wi-Fi).

## API

| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| `GET` | `/` | — | Chat UI (HTML) |
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/ask` | `{"question": "..."}` | `{"answer": "..."}` |
| `POST` | `/reset` | — | Clears conversation history |

## Audio Processing

Each WAV file is processed through a 4-stage pipeline:

1. **Load & normalize** — native sample rate, mono, amplitude normalized to `[-1, 1]`
2. **Feature extraction** — 10 features:

| Feature | Fault indicator |
|---------|----------------|
| RMS | Energy / load level |
| Kurtosis | Impulsive spikes — bearing/gear faults |
| Crest Factor | Peak impacts |
| Dominant Frequency | Speed / resonance shifts |
| Spectral Centroid | Friction / wear (rises with degradation) |
| Spectral Bandwidth | Broadband faults |
| Low-band Energy Ratio | Energy migration (healthy: > 45% in 80–500 Hz) |
| Harmonic Ratio | Looseness / noise (drops below 0.3) |
| Zero-Crossing Rate | Chattering / high-freq noise |
| MFCC Delta | Spectral instability |

3. **Interpretation** — features mapped to plain-English observations + severity (`normal` / `warning` / `fault`)
4. **RAG chunk** — all features + observations serialized as a text document for vector embedding

Filename format: `YYYYMMDD_HHMMSSxxxxxxZ_mb6000_sensorN.wav`
