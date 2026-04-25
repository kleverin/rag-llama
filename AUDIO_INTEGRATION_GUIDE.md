# Audio Processing Integration Guide
## MB6000 Acoustic Pipeline — For Pipeline Integration

---

## Overview

This guide explains how the three audio-related scripts fit into the RAG pipeline and how to integrate them on your end. The audio pipeline converts raw `.wav` recordings from the MB6000 sensor into text chunks that get embedded into ChromaDB alongside MTConnect data and the MB4000 troubleshooting knowledge base.

```
MB6000 sensor
     |
     v
.wav file (e.g. 20250528_035938.263732Z_mb6000_sensor1.wav)
     |
     v
audio_processing.py  <-- extracts 10 acoustic features + narrative
     |
     v
ingest.py  <-- embeds into ChromaDB with MTConnect + knowledge base
     |
     v
ChromaDB vector store
     |
     v
server.py  <-- FastAPI serves queries via /ask
     |
     v
Llama 3.2 (via Ollama) generates diagnosis
```

---

## File Summary

| File | Role | When to run |
|------|------|-------------|
| `audio_processing.py` | Extracts features from a single `.wav` file, returns feature dict + RAG text chunk | Called by `ingest.py` automatically, or run standalone for testing |
| `ingest.py` | Walks your data folder, processes all `.wav` + `.xlsx` + `.md` files, builds ChromaDB index | Run once before starting server, and re-run whenever new data arrives |
| `server.py` | FastAPI server that answers questions using the ChromaDB index | Run after ingest, keep running |
| `mb4000_troubleshooting.md` | Diagnostic knowledge base for the MB4000 | Sits in project root, loaded automatically by `ingest.py` |

---

## Prerequisites

### Python packages
All packages must be installed in the `iiot/` virtual environment:

```bash
source iiot/bin/activate

pip install librosa scipy numpy
pip install llama-index llama-index-vector-stores-chroma
pip install llama-index-embeddings-ollama llama-index-llms-ollama
pip install chromadb fastapi uvicorn python-dotenv
pip install pandas openpyxl
```

### Ollama models
Ollama must be running with both models pulled before anything else:

```bash
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text
```

### `.env` file
Create a `.env` file in the project root:

```
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PATH=./chroma_db
DATA_PATH=./data
KB_PATH=./
PORT=8000
```

`KB_PATH` points to the folder containing `mb4000_troubleshooting.md` — keep it in the project root (same folder as the scripts).

---

## Project Folder Structure

```
project_root/
├── audio_processing.py         <- acoustic feature extraction
├── ingest.py                   <- data ingestion pipeline
├── server.py                   <- FastAPI query server
├── mb4000_troubleshooting.md   <- diagnostic knowledge base
├── .env                        <- config (do not commit to git)
├── requirements.txt            <- pip dependencies
├── chroma_db/                  <- auto-created by ingest.py
└── data/
    └── 2026_Spring_ME597/
        ├── KRPM_shift_report_Apr-June2025/
        │   ├── shift_report_001.xlsx
        │   └── ...
        ├── MTConnect_data/
        │   └── krpm.operation_20250901.xlsx
        └── audio/              <- put .wav files here (or any subfolder)
            ├── 20250528_035938.263732Z_mb6000_sensor1.wav
            └── 20250528_035838.274150Z_mb6000_sensor1.wav
```

The `.wav` files can be in any subfolder under `data/` — `ingest.py` walks the entire directory tree recursively.

---

## WAV File Naming Convention

The MB6000 sensor produces files in this format:

```
20250528_035938.263732Z_mb6000_sensor1.wav
^date    ^time           ^device ^sensor
```

`audio_processing.py` parses the timestamp directly from the filename. If your files have a different naming convention, update the `parse_timestamp_from_filename()` function in `audio_processing.py`.

---

## How audio_processing.py Works

### What it extracts
Each `.wav` file produces 10 acoustic features:

| Feature | What it detects |
|---------|----------------|
| RMS | Overall energy / loudness level |
| Kurtosis | Impulsive spikes — bearing and gear faults |
| Crest Factor | Peak-to-average ratio — confirms impact events |
| Dominant Frequency | Main frequency in Hz — correlates with shaft RPM |
| Spectral Centroid | Centre of mass of spectrum — rises with friction/wear |
| Spectral Bandwidth | Spread of energy — wide = broadband fault |
| Low-band Energy Ratio | Fraction of energy in 80-500 Hz — healthy machines concentrate energy here |
| Harmonic Ratio | Tonal vs noisy content — drops with looseness |
| Zero-Crossing Rate | High-frequency chattering or noise |
| Pitch Tracking | Mean, std, min, max pitch — detects speed instability |

### What it produces
Each processed file produces a **RAG text chunk** like this:

```
[AUDIO ANALYSIS] File: 20250528_035938.263732Z_mb6000_sensor1.wav |
Timestamp: 2025-05-28T03:59:38Z | Severity: WARNING

Acoustic features — RMS: 0.2841 | Kurtosis: 5.21 | Crest factor: 6.43 |
Fault flag: FAULT DETECTED
Dominant frequency: 247.3 Hz | Spectral centroid: 2841.2 Hz |
Low-band energy: 48.3% | Harmonic ratio: 0.41
Pitch — mean: 243.1 Hz, std: 41.2 Hz, range: 180.3-312.7 Hz
Zero-crossing rate: 0.0821 | Spectral bandwidth: 1923.4 Hz

Interpretation: Kurtosis (5.21) exceeds the fault threshold, indicating early-stage
impulsive events — possible bearing wear or developing gear damage. Monitor closely.
Pitch is unstable (std=41.2 Hz), suggesting speed fluctuation or load changes during
the recording window — possible tool wear or intermittent contact.
```

This text is what gets embedded into ChromaDB and retrieved when someone asks a question.

### Severity levels
Every processed file gets a severity label:

| Severity | Condition | Recommended action |
|----------|-----------|-------------------|
| `normal` | All features within healthy ranges | No action |
| `warning` | One or more features approaching thresholds | Monitor, plan inspection |
| `fault` | Kurtosis > 4.0 AND crest factor > 5.0 | Inspect soon |

### Adjusting thresholds
The `THRESHOLDS` dict at the top of `audio_processing.py` controls all decision boundaries. Tune these after you have a healthy baseline dataset from your specific machine:

```python
THRESHOLDS = {
    "rms_high":               0.35,   # adjust based on your typical cutting loads
    "kurtosis_fault":         4.0,    # lower = more sensitive to bearing faults
    "crest_factor_fault":     5.0,    # lower = more sensitive to impacts
    "spectral_centroid_high": 3500,   # raise if your machine normally runs at high freq
    ...
}
```

---

## How ingest.py Works

### What it ingests
`ingest.py` walks `DATA_PATH` and processes three types of data in order:

1. **Knowledge base** (`.md` files in `KB_PATH`) — loaded first so diagnostic context always has high priority
2. **MTConnect data** (`.xlsx` files) — each row becomes one document with natural-language text + metadata
3. **Audio data** (`.wav` files) — each file is processed through `audio_processing.py` and embedded

### Running ingest

```bash
source iiot/bin/activate
python ingest.py
```

Expected output:
```
── Loading Knowledge Base ─────────────────────────────────
  [kb]  mb4000_troubleshooting.md — 10 sections
  -> 10 knowledge base chunks

[scan]  2 Excel files, 4 WAV files

── Ingesting Excel files ──────────────────────────────────
  [xlsx]  krpm.operation_20250901.xlsx — 1284 rows

── Ingesting WAV files ────────────────────────────────────
  [NORMAL ]  20250528_035838.274150Z_mb6000_sensor1.wav
  [WARNING]  20250528_035938.263732Z_mb6000_sensor1.wav
             -> Kurtosis (5.21) exceeds fault threshold...

[audio]  0 fault(s), 1 warning(s) found during ingest

── Building index (1296 documents) ────────────────────────
[done]  1296 documents embedded into ./chroma_db
```

### Re-ingesting after new data arrives
Delete `chroma_db/` and re-run to force a full rebuild:

```bash
rm -rf chroma_db/
python ingest.py
```

Or just run `python ingest.py` again — it will append to the existing collection.

---

## How server.py Works

### Starting the server

```bash
source iiot/bin/activate
python server.py
```

The server starts on port 8000. Access it from any device on the same network:

```
http://your-machine-ip:8000/        <- chat UI (browser, mobile-friendly)
http://your-machine-ip:8000/health  <- health check
http://your-machine-ip:8000/ask     <- POST endpoint
http://your-machine-ip:8000/reset   <- POST to clear conversation
```

### API usage

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Was there anything unusual in the sound data on May 28?"}'

# Clear conversation history
curl -X POST http://localhost:8000/reset
```

### Example questions and what the system does

| Question | What gets retrieved | Expected response type |
|----------|--------------------|-----------------------|
| "Was there any unusual sound on May 28 at 03:59?" | Audio chunk for that timestamp | Describes acoustic anomalies with severity |
| "Is the spindle bearing OK?" | Audio chunks + bearing fault section from knowledge base | Compares features against thresholds |
| "Why did the machine stop unexpectedly?" | MTConnect rows near estop event + alarm code section | Identifies load spike or E-stop cause |
| "The surface finish is getting worse" | Recent audio + tool wear section from knowledge base | Links spectral centroid trend to tool wear |
| "What does a bearing fault sound like?" | Bearing fault section from knowledge base | Describes acoustic signature |
| "Tool 13 has been running for 300 parts — is it OK?" | MTConnect partcount rows + tool life section | Cross-references load trend with part count |

---

## Integration Points for Your Pipeline

If you want to call `audio_processing.py` directly from your own code rather than through `ingest.py`, here is the interface:

### Process a single file

```python
from audio_processing import process_wav

result = process_wav("path/to/recording.wav")

# Access individual features
print(result["rms"])               # float
print(result["kurtosis"])          # float
print(result["severity"])          # 'normal' | 'warning' | 'fault'
print(result["fault_flag"])        # bool
print(result["observations"])      # list of plain-English strings
print(result["rag_chunk"])         # full text chunk — hand this to your embedder

# Access pitch details
print(result["pitch"]["mean"])     # float Hz
print(result["pitch"]["std"])      # float Hz — high = speed instability
```

### Process a whole directory

```python
from audio_processing import process_wav_directory

results = process_wav_directory("path/to/wav/folder/")

for r in results:
    if r["severity"] == "fault":
        print(f"FAULT in {r['file']} at {r['timestamp']}")
        for obs in r["observations"]:
            print(f"  {obs}")
```

### Full feature dict keys

```python
{
    "file":                   str,    # filename
    "timestamp":              str,    # ISO timestamp parsed from filename
    "rms":                    float,
    "kurtosis":               float,
    "crest_factor":           float,
    "dominant_freq_hz":       float,
    "spectral_centroid_hz":   float,
    "spectral_bandwidth_hz":  float,
    "low_band_energy_ratio":  float,  # 0.0 to 1.0
    "zero_crossing_rate":     float,
    "mfcc_delta_mean":        float,
    "harmonic_ratio":         float,  # 0.0 to 1.0
    "pitch": {
        "mean":         float,
        "std":          float,
        "min":          float,
        "max":          float,
        "voiced_ratio": float,        # fraction of frames with detectable pitch
    },
    "fault_flag":             bool,
    "severity":               str,    # 'normal' | 'warning' | 'fault'
    "observations":           list,   # plain-English interpretation strings
    "rag_chunk":              str,    # full embeddable text chunk
}
```

---

## Testing the Audio Pipeline Standalone

Before running the full ingest, test a single WAV file:

```bash
source iiot/bin/activate
python audio_processing.py path/to/your/recording.wav
```

Expected output:
```
[load]  20250528_035938.263732Z_mb6000_sensor1.wav | SR=48000 Hz | 1.00s

── Features ──────────────────────────────────────────────
  file                         20250528_035938.263732Z_mb6000_sensor1.wav
  timestamp                    2025-05-28T03:59:38Z
  rms                          0.2841
  kurtosis                     5.21
  crest_factor                 6.43
  dominant_freq_hz             247.3
  spectral_centroid_hz         2841.2
  spectral_bandwidth_hz        1923.4
  low_band_energy_ratio        0.483
  zero_crossing_rate           0.0821
  harmonic_ratio               0.41
  fault_flag                   True
  severity                     warning
  pitch:
    mean                       243.10
    std                        41.20
    min                        180.30
    max                        312.70
    voiced_ratio               0.87

── Observations ──────────────────────────────────────────
  -> Kurtosis (5.21) exceeds the fault threshold...
  -> Pitch is unstable (std=41.2 Hz)...

── RAG Chunk ─────────────────────────────────────────────
[AUDIO ANALYSIS] File: ...
```

Also test a directory:

```bash
python audio_processing.py path/to/wav/folder/
```

---

## Troubleshooting Common Issues

### librosa not found
```bash
source iiot/bin/activate
pip install librosa
```

### pyin pitch extraction fails or is slow
`pyin` is the most CPU-intensive step. It can be disabled by commenting out the call in `process_wav()` and replacing with a default return. This will make pitch features unavailable but everything else will still work:

```python
# In process_wav(), replace:
pitch = extract_pitch_series(y, sr)
# With:
pitch = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "voiced_ratio": 0.0}
```

### Ollama connection refused
Make sure Ollama is running before starting ingest or server:
```bash
ollama serve
```

### ChromaDB collection already exists error
Delete the old collection and re-ingest:
```bash
rm -rf chroma_db/
python ingest.py
```

### WAV file timestamp not parsed correctly
If your filenames use a different format than `20250528_035938.263732Z_mb6000_sensor1.wav`,
update the `parse_timestamp_from_filename()` function in `audio_processing.py`:

```python
def parse_timestamp_from_filename(filename: str) -> str:
    # Modify this to match your actual filename format
    ...
```

---

## Quick Start Checklist

```
[ ] Virtual environment activated: source iiot/bin/activate
[ ] Ollama running: ollama serve
[ ] Both models pulled: ollama pull llama3.2 && ollama pull nomic-embed-text
[ ] .env file created with correct paths
[ ] mb4000_troubleshooting.md in project root
[ ] WAV files placed under data/ (any subfolder)
[ ] xlsx files placed under data/ (any subfolder)
[ ] python ingest.py   -- runs clean, shows document count
[ ] python server.py   -- starts on port 8000
[ ] http://localhost:8000/health returns {"status": "ok"}
[ ] Ask a test question via browser or curl
```
