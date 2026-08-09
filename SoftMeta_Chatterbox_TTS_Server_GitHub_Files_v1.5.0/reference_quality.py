from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def decode_audio_mono(path: Path, sample_rate: int = 16000) -> np.ndarray:
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "pipe:1",
        ],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def analyze_reference_voice(path: Path) -> dict[str, Any]:
    """Fast acoustic screening before a clone reference reaches Chatterbox."""
    try:
        audio = decode_audio_mono(path, 16000)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"score": 0, "rating": "Unreadable", "warnings": [f"Could not decode reference audio: {exc}"], "usable": False}
    if audio.size == 0:
        return {"score": 0, "rating": "Unreadable", "warnings": ["Reference audio is empty."], "usable": False}

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    duration = audio.size / 16000.0
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))
    peak_db = 20 * math.log10(max(peak, 1e-6))
    rms_db = 20 * math.log10(max(rms, 1e-6))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.985))

    frame = 320
    count = max(1, audio.size // frame)
    frames = audio[:count * frame].reshape(count, frame)
    frms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    fdb = 20 * np.log10(np.maximum(frms, 1e-6))
    speech_ref = float(np.percentile(fdb, 80))
    noise_ref = float(np.percentile(fdb, 20))
    snr_proxy = speech_ref - noise_ref
    silence_threshold = min(-42.0, speech_ref - 28.0)
    silence_ratio = float(np.mean(fdb <= silence_threshold))
    dynamic_range = float(np.percentile(fdb, 90) - np.percentile(fdb, 10))

    score = 100.0
    warnings: list[str] = []
    if duration < 5.5:
        score -= 35; warnings.append("Reference is too short. Use at least 6 seconds; 10–20 seconds is preferred.")
    elif duration < 8.0:
        score -= 12; warnings.append("Reference is usable but a clean 10–20 second clip is more reliable.")
    elif duration > 35:
        score -= 6; warnings.append("Reference is longer than needed; a focused 10–20 second clean sample is preferable.")
    if rms_db < -32:
        score -= 24; warnings.append("Reference level is very low. Use a louder clean recording.")
    elif rms_db < -26:
        score -= 10; warnings.append("Reference level is a little quiet.")
    if peak_db > -0.15 or clipping_ratio > 0.001:
        score -= 22; warnings.append("Reference contains clipping or near-clipping peaks.")
    if silence_ratio > 0.40:
        score -= 18; warnings.append("Reference contains too much silence or dead air.")
    elif silence_ratio > 0.28:
        score -= 8; warnings.append("Reference contains more silence than ideal.")
    if snr_proxy < 14:
        score -= 22; warnings.append("Reference may contain room noise, echo, or weak speech separation.")
    elif snr_proxy < 20:
        score -= 8; warnings.append("Reference cleanliness is only moderate; a quieter recording may clone better.")
    if dynamic_range < 8:
        score -= 6; warnings.append("Reference dynamics are unusually flat or heavily processed.")

    score = int(round(max(0, min(100, score))))
    rating = "Excellent" if score >= 90 else "Good" if score >= 78 else "Fair" if score >= 62 else "Poor"
    return {
        "score": score, "rating": rating, "usable": score >= 50,
        "duration_seconds": round(duration, 2), "rms_dbfs": round(rms_db, 1),
        "peak_dbfs": round(peak_db, 1), "clipping_percent": round(clipping_ratio * 100, 3),
        "silence_percent": round(silence_ratio * 100, 1), "snr_proxy_db": round(snr_proxy, 1),
        "dynamic_range_db": round(dynamic_range, 1), "warnings": warnings,
    }
