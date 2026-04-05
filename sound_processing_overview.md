"""
# MB4000 — Audio Feature Extraction

## Overview
This module is part of the **AI Failure Mode Analysis** pipeline.

FIND THE SAMPLE FLOW OF TASKS BELOW:

### 1. Task (Sound Processing)
- Load raw WAV recordings from the MB4000 sensor
- Extract 6 core features that describe machine health
- Serialize features into a text chunk for the RAG pipeline

### 2. Pipeline Position
```
WAV file --> [audio_processing.py] --> text chunk --> RAG vector store --> Llama 3.2
```

### 3. Dependencies
```bash
pip install librosa numpy scipy
```

### 4. Usage
```bash
# From command line
python audio_processing.py path/to/sensor_recording.wav

# In your RAG pipeline
from audio_processing import process_wav
result = process_wav("20251105_123906_mb4000_sensor1.wav")
rag_chunk = result["rag_chunk"]   # hand this to your embedder
```
"""

import os
import sys
import json
import numpy as np
import librosa
from scipy.stats import kurtosis


# ─────────────────────────────────────────────────────────
# STAGE 1 — LOAD & PREPROCESS

def load_wav(filepath: str) -> tuple:
    """
    Load a WAV file from disk.

    **Args**
    - `filepath` — path to the `.wav` file

    **Returns**
    - `y` — audio time series as a numpy array
    - `sr` — sample rate in Hz (e.g. 48000)

    **Notes**
    - `mono=True` collapses stereo to a single channel
    - `sr=None` preserves the native sample rate — do NOT resample,
      frequency features depend on SR being correct
    """
    y, sr = librosa.load(filepath, sr=None, mono=True)
    print(f"[load]  {os.path.basename(filepath)} | SR={sr}Hz | {len(y)/sr:.1f}s")
    return y, sr


def normalize(y: np.ndarray) -> np.ndarray:
    """
    Normalize audio amplitude to the `[-1, 1]` range.

    **Why:** Recordings captured on different days at slightly different
    gain levels would produce different RMS values even for identical
    machine behavior. Normalization removes that variation so features
    are comparable across files.
    """
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y  # BUG FIX: original was missing this return


# ─────────────────────────────────────────────────────────
# STAGE 2 — FEATURE EXTRACTION

def extract_rms(y: np.ndarray) -> float:
    """
    Root Mean Square amplitude — the overall energy level.

    **What it tells you**
    - Higher RMS = louder / more energetic machine operation
    - A sudden drop or spike vs baseline can indicate a fault

    **Formula:** `sqrt(mean(y^2))`
    """
    return float(np.sqrt(np.mean(y ** 2)))


def extract_kurtosis(y: np.ndarray) -> float:
    """
    Statistical kurtosis of the signal — measures 'spikiness'.

    **What it tells you**

    | Value | Meaning |
    |-------|---------|
    | ~3    | Normal Gaussian signal |
    | > 4   | Impulsive spikes present (early fault signal) |
    | > 10  | Strong fault signature (e.g. bearing defect) |

    **Why it matters for MB4000:**
    Bearing faults and gear tooth damage produce sharp periodic impacts.
    Kurtosis catches these early, even before they are audible to a human.
    """
    return float(kurtosis(y))


def extract_crest_factor(y: np.ndarray) -> float:
    """
    Ratio of peak amplitude to RMS amplitude.

    **What it tells you**
    - Low crest factor (~1.4 for a sine wave) → smooth, periodic signal
    - Crest factor > 5 → impulsive events present
    - Used alongside kurtosis to confirm fault signatures

    **Formula:** `peak / RMS`
    """
    rms = extract_rms(y)
    peak = float(np.max(np.abs(y)))
    return peak / (rms + 1e-10)  # epsilon avoids division by zero


def extract_dominant_frequency(y: np.ndarray, sr: int) -> float:
    """
    The frequency (Hz) carrying the most energy in the signal.

    **What it tells you**
    For a rotating machine the dominant frequency is usually the shaft
    rotation frequency or one of its harmonics. A shift from baseline can
    indicate:
    - Speed change (load variation)
    - Looseness (frequency drops)
    - Resonance (frequency locks to a structural mode)

    > Example: MB4000 recording shows ~164 Hz dominant —
    > likely corresponding to the machine's shaft RPM.
    """
    fft_magnitude = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    dominant_idx = np.argmax(fft_magnitude)
    return float(freqs[dominant_idx])


def extract_low_band_energy_ratio(y: np.ndarray, sr: int) -> float:
    """
    Fraction of total spectral energy in the **80–500 Hz** band.

    **What it tells you**
    Most rotating machinery faults (imbalance, misalignment, looseness)
    manifest as low-frequency energy increases. A healthy MB4000 should
    have most energy concentrated in this band.

    If energy shifts upward into mid/high bands it may indicate bearing
    wear or friction developing.

    **Returns:** a ratio between `0.0` and `1.0`
    > e.g. `0.71` means 71% of energy is in 80–500 Hz — healthy
    """
    stft = np.abs(librosa.stft(y)) ** 2        # power spectrogram
    freqs = librosa.fft_frequencies(sr=sr)

    low_mask = (freqs >= 80) & (freqs < 500)   # band of interest
    total_energy = np.sum(stft) + 1e-10
    low_energy = np.sum(stft[low_mask, :])

    return float(low_energy / total_energy)


def compute_fault_flag(kurtosis_val: float, crest_factor: float) -> bool:
    """
    Simple rule-based fault indicator.

    **Triggers when BOTH conditions are true:**
    - `kurtosis > 4` — impulsive signal
    - `crest_factor > 5` — high peak-to-average ratio

    Both conditions together strongly suggest impulsive mechanical events
    such as a bearing fault, gear damage, or mechanical knock.
    A single condition alone is prone to false positives.

    **Returns**
    - `True` → possible fault detected
    - `False` → signal looks normal
    """
    return kurtosis_val > 4.0 and crest_factor > 5.0


# ─────────────────────────────────────────────────────────
# STAGE 3 — SERIALIZE TO RAG TEXT CHUNK

def parse_timestamp_from_filename(filename: str) -> str:
    """
    Extract a human-readable timestamp from the MB4000 filename format.

    **Filename format:**
    ```
    20251105_123906_066025Z_mb4000_sensor1.wav
    ^date    ^time
    ```

    **Returns:** `'2025-11-05 12:39:06 UTC'`
    """
    try:
        base = os.path.basename(filename).replace(".wav", "")
        parts = base.split("_")
        date_str = parts[0]   # '20251105'
        time_str = parts[1]   # '123906'
        date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]} UTC"
        return f"{date} {time}"
    except Exception:
        return "timestamp unknown"


def to_rag_chunk(features: dict) -> str:
    """
    Convert the feature dict into a plain-text chunk for RAG ingestion.

    This string is what gets embedded into the vector store.
    Written in natural language so the embedding model can match it
    semantically against user queries such as:
    - *"any faults on Nov 5 day shift?"*
    - *"was the machine running normally?"*

    **Args**
    - `features` — dict returned by `process_wav()`

    **Returns**
    - A short readable string describing the recording

    **Example output:**
    ```
    Sensor recording: 20251105_123906_mb4000_sensor1.wav | Timestamp: 2025-11-05 12:39:06 UTC
    RMS: 0.1947 | Kurtosis: 0.17 | Crest factor: 5.14
    Dominant frequency: 160.7 Hz | Low-band energy: 71.0%
    Fault flag: none
    ```
    """
    flag_str = "FAULT DETECTED" if features["fault_flag"] else "none"

    chunk = (
        f"Sensor recording: {features['file']} | "
        f"Timestamp: {features['timestamp']}\n"
        f"RMS: {features['rms']:.4f} | "
        f"Kurtosis: {features['kurtosis']:.2f} | "
        f"Crest factor: {features['crest_factor']:.2f}\n"
        f"Dominant frequency: {features['dominant_freq_hz']:.1f} Hz | "
        f"Low-band energy: {features['low_band_energy_ratio']*100:.1f}%\n"
        f"Fault flag: {flag_str}"
    )
    return chunk


# ─────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION

def process_wav(filepath: str) -> dict:
    """
    Full pipeline for one WAV file — runs all 3 stages.

    ```
    Stage 1: load_wav → normalize
    Stage 2: extract rms, kurtosis, crest_factor,
             dominant_freq, low_band_energy, fault_flag
    Stage 3: serialize → rag_chunk string
    ```

    **Args**
    - `filepath` — path to the `.wav` file

    **Returns** a dict with keys:

    | Key | Type | Description |
    |-----|------|-------------|
    | `file` | str | filename |
    | `timestamp` | str | parsed from filename |
    | `rms` | float | overall energy |
    | `kurtosis` | float | signal spikiness |
    | `crest_factor` | float | peak-to-average ratio |
    | `dominant_freq_hz` | float | main frequency in Hz |
    | `low_band_energy_ratio` | float | 0.0–1.0 |
    | `fault_flag` | bool | True = possible fault |
    | `rag_chunk` | str | **hand this to the RAG pipeline** |
    """
    # Stage 1 — Load & preprocess
    y, sr = load_wav(filepath)
    y = normalize(y)

    # Stage 2 — Extract features
    rms        = extract_rms(y)
    kurt       = extract_kurtosis(y)
    crest      = extract_crest_factor(y)
    dom_freq   = extract_dominant_frequency(y, sr)
    low_energy = extract_low_band_energy_ratio(y, sr)
    fault      = compute_fault_flag(kurt, crest)

    features = {
        "file":                  os.path.basename(filepath),
        "timestamp":             parse_timestamp_from_filename(filepath),
        "rms":                   rms,
        "kurtosis":              kurt,
        "crest_factor":          crest,
        "dominant_freq_hz":      dom_freq,
        "low_band_energy_ratio": low_energy,
        "fault_flag":            fault,
    }

    # Stage 3 — Serialize to RAG text chunk
    features["rag_chunk"] = to_rag_chunk(features)

    return features


# ─────────────────────────────────────────────────────────
# RUN FROM COMMAND LINE

if __name__ == "__main__":
    wav_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/user-data/uploads/20251105_123906_066025Z_mb6000_sensor1.wav"

    result = process_wav(wav_path)

    print("\n── Features ──────────────────────────────")
    for k, v in result.items():
        if k != "rag_chunk":
            print(f"  {k:<28} {v}")

    print("\n── RAG Chunk (hand this to RAG pipeline) ─")
    print(result["rag_chunk"])

    # Save JSON output
    out_dir = "/mnt/user-data/outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, result["file"].replace(".wav", "_features.json"))
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {out_path}")
