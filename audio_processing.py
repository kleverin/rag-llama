"""
audio_processing.py — MB6000 Audio Feature Extraction
======================================================
Part of the AI Failure Mode Analysis RAG pipeline.

Pipeline position:
    WAV file --> [audio_processing.py] --> text chunk --> RAG vector store --> Llama 3.2

Filename format expected:
    20250528_035938.263732Z_mb6000_sensor1.wav
    ^date    ^time           ^device ^sensor

Usage:
    python audio_processing.py path/to/recording.wav
    from audio_processing import process_wav
"""

import os
import sys
import json
import numpy as np
import librosa
import librosa.feature
from scipy.stats import kurtosis as scipy_kurtosis


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE THRESHOLDS — tuned for MB6000
# Adjust these after you have a healthy baseline dataset
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "rms_high":              0.35,    # above = louder than normal operation
    "rms_low":               0.03,    # below = very quiet / possibly off
    "kurtosis_fault":        4.0,     # above = impulsive spikes (bearing/gear fault)
    "kurtosis_severe":       10.0,    # above = severe fault signature
    "crest_factor_fault":    5.0,     # above = high peak-to-average (impact events)
    "spectral_centroid_high": 3500,   # Hz — above = high-freq energy (friction/wear)
    "spectral_centroid_low":  300,    # Hz — below = mostly low-freq (imbalance)
    "low_band_healthy_min":  0.45,    # 80-500 Hz should hold at least 45% energy
    "pitch_spike_ratio":     1.4,     # dominant freq > baseline * this = pitch spike
    "pitch_drop_ratio":      0.65,    # dominant freq < baseline * this = pitch drop
    "zcr_high":              0.15,    # zero-crossing rate — high = noisy/chattering
}

# Expected dominant frequency range for healthy MB6000 operation (Hz)
HEALTHY_FREQ_RANGE = (80, 600)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def load_wav(filepath: str) -> tuple:
    """Load WAV, preserve native SR, collapse to mono."""
    y, sr = librosa.load(filepath, sr=None, mono=True)
    print(f"[load]  {os.path.basename(filepath)} | SR={sr} Hz | {len(y)/sr:.2f}s")
    return y, sr


def normalize(y: np.ndarray) -> np.ndarray:
    """Normalize amplitude to [-1, 1]."""
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_rms(y: np.ndarray) -> float:
    """Overall signal energy — sqrt(mean(y^2))."""
    return float(np.sqrt(np.mean(y ** 2)))


def extract_kurtosis(y: np.ndarray) -> float:
    """
    Statistical kurtosis — measures impulsive spikes.
    ~3 = normal Gaussian | >4 = early fault | >10 = severe fault
    """
    return float(scipy_kurtosis(y))


def extract_crest_factor(y: np.ndarray) -> float:
    """Peak-to-RMS ratio. >5 indicates impulsive mechanical events."""
    rms  = extract_rms(y)
    peak = float(np.max(np.abs(y)))
    return peak / (rms + 1e-10)


def extract_dominant_frequency(y: np.ndarray, sr: int) -> float:
    """Frequency bin carrying the most spectral energy (Hz)."""
    fft_mag = np.abs(np.fft.rfft(y))
    freqs   = np.fft.rfftfreq(len(y), d=1.0 / sr)
    return float(freqs[np.argmax(fft_mag)])


def extract_spectral_centroid(y: np.ndarray, sr: int) -> float:
    """
    Weighted mean frequency — the 'centre of mass' of the spectrum.
    Rises when wear or friction shifts energy toward higher frequencies.
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(np.mean(centroid))


def extract_spectral_bandwidth(y: np.ndarray, sr: int) -> float:
    """
    Spread of energy around the spectral centroid.
    Healthy machines tend to have narrower bandwidth (concentrated energy).
    """
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return float(np.mean(bw))


def extract_low_band_energy_ratio(y: np.ndarray, sr: int,
                                   low=80, high=500) -> float:
    """
    Fraction of total energy in the 80-500 Hz band.
    Healthy rotating machinery: most energy lives here.
    Energy migration upward suggests bearing wear or friction.
    """
    stft       = np.abs(librosa.stft(y)) ** 2
    freqs      = librosa.fft_frequencies(sr=sr)
    low_mask   = (freqs >= low) & (freqs < high)
    total      = np.sum(stft) + 1e-10
    low_energy = np.sum(stft[low_mask, :])
    return float(low_energy / total)


def extract_zero_crossing_rate(y: np.ndarray) -> float:
    """
    Rate at which the signal changes sign.
    High ZCR = noisy, chattering, or high-frequency content.
    """
    zcr = librosa.feature.zero_crossing_rate(y)
    return float(np.mean(zcr))


def extract_mfcc_delta(y: np.ndarray, sr: int, n_mfcc: int = 13) -> float:
    """
    Mean absolute value of MFCC first-order delta coefficients.
    Captures how rapidly the spectral shape is changing — elevated
    values suggest transient or unstable machine operation.
    """
    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    return float(np.mean(np.abs(mfcc_delta)))


def extract_harmonic_ratio(y: np.ndarray) -> float:
    """
    Ratio of harmonic (tonal) to total energy.
    Healthy rotating machinery is highly harmonic.
    A drop suggests noise, looseness, or irregular motion.
    """
    y_harmonic, _ = librosa.effects.hpss(y)
    harmonic_energy = float(np.sum(y_harmonic ** 2))
    total_energy    = float(np.sum(y ** 2)) + 1e-10
    return harmonic_energy / total_energy


def extract_pitch_series(y: np.ndarray, sr: int) -> dict:
    """
    Track instantaneous pitch over time using pyin.
    Returns mean, std, min, max of voiced pitch values (Hz).
    High std or sudden shifts indicate speed/load instability.
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=2000, sr=sr,
            frame_length=2048
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        if len(voiced_f0) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "voiced_ratio": 0.0}
        return {
            "mean":         float(np.mean(voiced_f0)),
            "std":          float(np.std(voiced_f0)),
            "min":          float(np.min(voiced_f0)),
            "max":          float(np.max(voiced_f0)),
            "voiced_ratio": float(len(voiced_f0) / len(f0)),
        }
    except Exception:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "voiced_ratio": 0.0}


def compute_fault_flag(kurtosis_val: float, crest_factor: float) -> bool:
    """Both kurtosis >4 AND crest factor >5 together = possible fault."""
    return (kurtosis_val > THRESHOLDS["kurtosis_fault"] and
            crest_factor > THRESHOLDS["crest_factor_fault"])


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — NARRATIVE INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

def interpret_features(features: dict) -> dict:
    """
    Convert numeric features into human-readable observations.
    Returns a dict of observations and a severity level.

    Severity levels: 'normal' | 'warning' | 'fault'
    """
    observations = []
    severity     = "normal"
    T            = THRESHOLDS

    rms      = features["rms"]
    kurt     = features["kurtosis"]
    crest    = features["crest_factor"]
    dom_freq = features["dominant_freq_hz"]
    centroid = features["spectral_centroid_hz"]
    low_band = features["low_band_energy_ratio"]
    zcr      = features["zero_crossing_rate"]
    pitch    = features["pitch"]
    harmonic = features["harmonic_ratio"]

    # ── Energy level ─────────────────────────────────────────────────────
    if rms > T["rms_high"]:
        observations.append(
            f"Signal energy (RMS={rms:.3f}) is elevated above the normal threshold "
            f"({T['rms_high']}), suggesting the machine was under higher load or "
            f"experiencing mechanical resistance."
        )
        severity = "warning"
    elif rms < T["rms_low"]:
        observations.append(
            f"Signal energy (RMS={rms:.3f}) is unusually low — the machine may have "
            f"been idle, or the sensor lost contact."
        )

    # ── Impulsive fault indicators ────────────────────────────────────────
    if kurt > T["kurtosis_severe"]:
        observations.append(
            f"Kurtosis is severely elevated ({kurt:.1f} > {T['kurtosis_severe']}). "
            f"This level of impulsive spikiness strongly indicates a mechanical fault "
            f"such as a bearing defect, gear tooth damage, or a mechanical knock. "
            f"Immediate inspection is recommended."
        )
        severity = "fault"
    elif kurt > T["kurtosis_fault"]:
        observations.append(
            f"Kurtosis ({kurt:.1f}) exceeds the fault threshold ({T['kurtosis_fault']}), "
            f"indicating early-stage impulsive events — possible bearing wear or "
            f"developing gear damage. Monitor closely."
        )
        if severity != "fault":
            severity = "warning"

    if crest > T["crest_factor_fault"]:
        observations.append(
            f"Crest factor ({crest:.1f}) is high, confirming sharp impact events in the "
            f"signal. Combined with elevated kurtosis this is a strong fault indicator."
            if kurt > T["kurtosis_fault"] else
            f"Crest factor ({crest:.1f}) is elevated above {T['crest_factor_fault']}, "
            f"suggesting occasional peak impacts even if the overall signal looks normal."
        )
        if severity == "normal":
            severity = "warning"

    # ── Pitch / dominant frequency ────────────────────────────────────────
    if dom_freq < HEALTHY_FREQ_RANGE[0]:
        observations.append(
            f"Dominant frequency ({dom_freq:.1f} Hz) is below the expected healthy range "
            f"({HEALTHY_FREQ_RANGE[0]}-{HEALTHY_FREQ_RANGE[1]} Hz). "
            f"This may indicate speed reduction, looseness, or low-frequency resonance."
        )
        if severity == "normal":
            severity = "warning"
    elif dom_freq > HEALTHY_FREQ_RANGE[1]:
        observations.append(
            f"Dominant frequency ({dom_freq:.1f} Hz) is above the expected healthy range. "
            f"High-frequency dominance often points to friction, bearing wear, or a "
            f"resonance condition."
        )
        if severity == "normal":
            severity = "warning"

    if pitch["std"] > 30:
        observations.append(
            f"Pitch is unstable (std={pitch['std']:.1f} Hz, mean={pitch['mean']:.1f} Hz). "
            f"Significant pitch variation suggests speed fluctuation or load changes "
            f"during the recording window — possible tool wear or intermittent contact."
        )
        if severity == "normal":
            severity = "warning"

    if pitch["mean"] > 0 and pitch["std"] / (pitch["mean"] + 1e-5) > 0.3:
        observations.append(
            f"The pitch-to-mean ratio is high, indicating the machine was not running "
            f"at a consistent speed during this window."
        )

    # ── Spectral character ────────────────────────────────────────────────
    if centroid > T["spectral_centroid_high"]:
        observations.append(
            f"Spectral centroid ({centroid:.0f} Hz) is high — energy has shifted toward "
            f"higher frequencies. This pattern is consistent with surface wear, "
            f"friction, or a cutting tool operating outside its design envelope."
        )
        if severity == "normal":
            severity = "warning"
    elif centroid < T["spectral_centroid_low"]:
        observations.append(
            f"Spectral centroid ({centroid:.0f} Hz) is very low — the signal is dominated "
            f"by low-frequency content, which may indicate rotor imbalance or structural "
            f"resonance."
        )

    if low_band < T["low_band_healthy_min"]:
        observations.append(
            f"Low-band energy ratio ({low_band*100:.1f}%) is below the healthy minimum "
            f"({T['low_band_healthy_min']*100:.0f}%). Energy has migrated to higher "
            f"frequency bands, which can precede bearing or gear failures."
        )
        if severity == "normal":
            severity = "warning"

    if harmonic < 0.3:
        observations.append(
            f"Harmonic content ratio is low ({harmonic:.2f}). A healthy rotating machine "
            f"should produce predominantly tonal (harmonic) sound. Low harmonic ratio "
            f"suggests noise, looseness, or irregular mechanical contact."
        )
        if severity == "normal":
            severity = "warning"

    # ── Zero crossing rate ────────────────────────────────────────────────
    if zcr > T["zcr_high"]:
        observations.append(
            f"Zero-crossing rate ({zcr:.3f}) is elevated, consistent with high-frequency "
            f"noise or chattering — possibly related to tool vibration or coolant flow "
            f"interference with the sensor."
        )

    # ── All normal ────────────────────────────────────────────────────────
    if not observations:
        observations.append(
            f"All acoustic features are within normal ranges. "
            f"RMS={rms:.3f}, kurtosis={kurt:.2f}, dominant freq={dom_freq:.1f} Hz. "
            f"No anomalies detected in this recording window."
        )

    return {
        "severity":     severity,
        "observations": observations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — SERIALIZE TO RAG TEXT CHUNK
# ─────────────────────────────────────────────────────────────────────────────

def parse_timestamp_from_filename(filename: str) -> str:
    """
    Parse timestamp from MB6000 filename format:
        20250528_035938.263732Z_mb6000_sensor1.wav
    Returns ISO string: '2025-05-28T03:59:38Z'
    """
    try:
        base  = os.path.basename(filename)
        parts = base.split("_")
        date_str = parts[0]                    # '20250528'
        time_raw = parts[1].split(".")[0]      # '035938' (drop microseconds)
        date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        time = f"{time_raw[:2]}:{time_raw[2:4]}:{time_raw[4:]}Z"
        return f"{date}T{time}"
    except Exception:
        return "timestamp-unknown"


def to_rag_chunk(features: dict) -> str:
    """
    Serialize features + narrative interpretation into a RAG text chunk.

    This is what gets embedded into ChromaDB. Written in natural language
    so Llama 3.2 can answer questions like:
    - "Was there any unusual sound on May 28 around 3:59 AM?"
    - "Did the pitch spike before the tool change at 06:05?"
    - "Any bearing fault indicators this week?"
    """
    flag_str     = "FAULT DETECTED" if features["fault_flag"] else "none"
    severity_str = features["severity"].upper()
    obs_text     = " ".join(features["observations"])
    pitch        = features["pitch"]

    chunk = (
        f"[AUDIO ANALYSIS] File: {features['file']} | "
        f"Timestamp: {features['timestamp']} | "
        f"Severity: {severity_str}\n"
        f"\n"
        f"Acoustic features — "
        f"RMS: {features['rms']:.4f} | "
        f"Kurtosis: {features['kurtosis']:.2f} | "
        f"Crest factor: {features['crest_factor']:.2f} | "
        f"Fault flag: {flag_str}\n"
        f"Dominant frequency: {features['dominant_freq_hz']:.1f} Hz | "
        f"Spectral centroid: {features['spectral_centroid_hz']:.1f} Hz | "
        f"Low-band energy: {features['low_band_energy_ratio']*100:.1f}% | "
        f"Harmonic ratio: {features['harmonic_ratio']:.2f}\n"
        f"Pitch — mean: {pitch['mean']:.1f} Hz, "
        f"std: {pitch['std']:.1f} Hz, "
        f"range: {pitch['min']:.1f}-{pitch['max']:.1f} Hz\n"
        f"Zero-crossing rate: {features['zero_crossing_rate']:.4f} | "
        f"Spectral bandwidth: {features['spectral_bandwidth_hz']:.1f} Hz\n"
        f"\n"
        f"Interpretation: {obs_text}"
    )
    return chunk


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def process_wav(filepath: str) -> dict:
    """
    Full pipeline for one WAV file.

    Returns a dict with all features, interpretation, and rag_chunk.
    Hand features['rag_chunk'] to your RAG embedder.
    """
    # Stage 1 — Load & normalize
    y, sr = load_wav(filepath)
    y     = normalize(y)

    # Stage 2 — Extract features
    rms      = extract_rms(y)
    kurt     = extract_kurtosis(y)
    crest    = extract_crest_factor(y)
    dom_freq = extract_dominant_frequency(y, sr)
    centroid = extract_spectral_centroid(y, sr)
    bw       = extract_spectral_bandwidth(y, sr)
    low_band = extract_low_band_energy_ratio(y, sr)
    zcr      = extract_zero_crossing_rate(y)
    mfcc_d   = extract_mfcc_delta(y, sr)
    harmonic = extract_harmonic_ratio(y)
    pitch    = extract_pitch_series(y, sr)
    fault    = compute_fault_flag(kurt, crest)

    features = {
        "file":                    os.path.basename(filepath),
        "timestamp":               parse_timestamp_from_filename(filepath),
        "rms":                     rms,
        "kurtosis":                kurt,
        "crest_factor":            crest,
        "dominant_freq_hz":        dom_freq,
        "spectral_centroid_hz":    centroid,
        "spectral_bandwidth_hz":   bw,
        "low_band_energy_ratio":   low_band,
        "zero_crossing_rate":      zcr,
        "mfcc_delta_mean":         mfcc_d,
        "harmonic_ratio":          harmonic,
        "pitch":                   pitch,
        "fault_flag":              fault,
    }

    # Stage 3 — Interpret features into narrative
    interpretation = interpret_features(features)
    features["severity"]     = interpretation["severity"]
    features["observations"] = interpretation["observations"]

    # Stage 4 — Serialize to RAG chunk
    features["rag_chunk"] = to_rag_chunk(features)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING — for ingest.py to call
# ─────────────────────────────────────────────────────────────────────────────

def process_wav_directory(directory: str) -> list:
    """
    Process all .wav files in a directory.
    Returns a list of feature dicts, one per file.
    Used by ingest.py for bulk ingestion.
    """
    results = []
    wav_files = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(".wav")
    ])

    if not wav_files:
        print(f"[audio]  No .wav files found in {directory}")
        return results

    print(f"[audio]  Found {len(wav_files)} WAV files in {directory}")
    for fname in wav_files:
        fpath = os.path.join(directory, fname)
        try:
            result = process_wav(fpath)
            results.append(result)
            print(f"  [{result['severity'].upper():7s}]  {fname}")
        except Exception as e:
            print(f"  [ERROR]   {fname} — {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_processing.py <path_to_wav_or_directory>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        results = process_wav_directory(target)
        print(f"\n[done]  Processed {len(results)} files")
        for r in results:
            print(f"\n{'─'*60}")
            print(f"File     : {r['file']}")
            print(f"Severity : {r['severity'].upper()}")
            for obs in r["observations"]:
                print(f"  -> {obs}")
    else:
        result = process_wav(target)
        print("\n── Features ──────────────────────────────────────────────")
        skip = {"rag_chunk", "observations", "pitch"}
        for k, v in result.items():
            if k not in skip:
                print(f"  {k:<28} {v}")
        print(f"\n  pitch:")
        for k, v in result["pitch"].items():
            print(f"    {k:<24} {v:.2f}")
        print("\n── Observations ──────────────────────────────────────────")
        for obs in result["observations"]:
            print(f"  -> {obs}")
        print("\n── RAG Chunk ─────────────────────────────────────────────")
        print(result["rag_chunk"])

        out_path = result["file"].replace(".wav", "_features.json")
        with open(out_path, "w") as f:
            json.dump(
                {k: v for k, v in result.items() if k != "rag_chunk"},
                f, indent=2
            )
        print(f"\n[saved]  {out_path}")
