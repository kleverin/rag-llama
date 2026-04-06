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
    Normalize audio amplitude to the [-1, 1] range.

    **Why:** Recordings captured on different days at slightly different
    gain levels would produce different RMS values even for identical
    machine behavior. Normalization removes that variation so features
    are comparable across files.
    """
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


# ─────────────────────────────────────────────────────────
# STAGE 2 — FEATURE EXTRACTION

def extract_rms(y: np.ndarray) -> float:
    """Root Mean Square amplitude — the overall energy level."""
    return float(np.sqrt(np.mean(y ** 2)))


def extract_kurtosis(y: np.ndarray) -> float:
    """Statistical kurtosis of the signal — measures 'spikiness'."""
    return float(kurtosis(y))


def extract_crest_factor(y: np.ndarray) -> float:
    """Ratio of peak amplitude to RMS amplitude."""
    rms = extract_rms(y)
    peak = float(np.max(np.abs(y)))
    return peak / (rms + 1e-10)  # epsilon avoids division by zero


def extract_dominant_frequency(y: np.ndarray, sr: int) -> float:
    """The frequency (Hz) carrying the most energy in the signal."""
    fft_magnitude = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    dominant_idx = np.argmax(fft_magnitude)
    return float(freqs[dominant_idx])


def extract_low_band_energy_ratio(y: np.ndarray, sr: int) -> float:
    """Fraction of total spectral energy in the 80–500 Hz band."""
    stft = np.abs(librosa.stft(y)) ** 2        # power spectrogram
    freqs = librosa.fft_frequencies(sr=sr)

    low_mask = (freqs >= 80) & (freqs < 500)   # band of interest
    total_energy = np.sum(stft) + 1e-10
    low_energy = np.sum(stft[low_mask, :])

    return float(low_energy / total_energy)


def compute_fault_flag(kurtosis_val: float, crest_factor: float) -> bool:
    """
    Simple rule-based fault indicator.
    Triggers when kurtosis > 4 AND crest_factor > 5.
    """
    return kurtosis_val > 4.0 and crest_factor > 5.0


# ─────────────────────────────────────────────────────────
# STAGE 3 — SERIALIZE TO RAG TEXT CHUNK

def parse_timestamp_from_filename(filename: str) -> str:
    """
    Extract a human-readable timestamp from the MB4000 filename format.

    Filename format: 20251105_123906_066025Z_mb4000_sensor1.wav
    Returns: '2025-11-05 12:39:06 UTC'
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
    """Convert the feature dict into a plain-text chunk for RAG ingestion."""
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

    Stage 1: load_wav → normalize
    Stage 2: extract rms, kurtosis, crest_factor, dominant_freq, low_band_energy, fault_flag
    Stage 3: serialize → rag_chunk string

    Returns a dict with keys: file, timestamp, rms, kurtosis, crest_factor,
    dominant_freq_hz, low_band_energy_ratio, fault_flag, rag_chunk.
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
    if len(sys.argv) < 2:
        print("Usage: python audio_processing.py path/to/sensor_recording.wav")
        sys.exit(1)

    result = process_wav(sys.argv[1])

    print("\n── Features ──────────────────────────────")
    for k, v in result.items():
        if k != "rag_chunk":
            print(f"  {k:<28} {v}")

    print("\n── RAG Chunk (hand this to RAG pipeline) ─")
    print(result["rag_chunk"])
