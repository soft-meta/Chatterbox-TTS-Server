from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

PRO_FINAL_TARGET_LUFS = -13.0
PRO_FINAL_TRUE_PEAK_DBFS = -1.0
PRO_FINAL_LRA = 7.0
VIDEO_MASTER_SAMPLE_RATE = 48000


def shape_professional_pauses(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_detect_ms: int = 650,
    medium_pause_ms: int = 390,
    long_pause_ms: int = 520,
    long_threshold_ms: int = 1200,
    leading_pause_ms: int = 65,
    trailing_pause_ms: int = 95,
) -> np.ndarray:
    """Keep breath-sized pauses but compact distracting dead air and generated hiss."""
    data = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
    if data.size == 0 or sample_rate <= 0:
        return data
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    frame_samples = max(1, int(round(sample_rate * 0.020)))
    frame_count = int(math.ceil(data.size / frame_samples))
    padded = np.pad(data, (0, frame_count * frame_samples - data.size))
    frames = padded.reshape(frame_count, frame_samples)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    speech_reference = float(np.percentile(db, 80))
    silence_threshold = float(np.clip(speech_reference - 27.0, -52.0, -38.0))
    silent = db <= silence_threshold
    min_frames = max(1, int(math.ceil(min_detect_ms / 20.0)))

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_silent in enumerate(silent):
        if is_silent and start is None:
            start = index
        elif not is_silent and start is not None:
            if index - start >= min_frames:
                runs.append((start, index))
            start = None
    if start is not None and frame_count - start >= min_frames:
        runs.append((start, frame_count))
    if not runs:
        return data

    parts: list[np.ndarray] = []
    cursor = 0
    edge_tolerance = frame_samples * 2
    for start_frame, end_frame in runs:
        start_sample = min(data.size, start_frame * frame_samples)
        end_sample = min(data.size, end_frame * frame_samples)
        if start_sample < cursor or end_sample <= start_sample:
            continue
        parts.append(data[cursor:start_sample])
        duration_ms = (end_sample - start_sample) * 1000.0 / sample_rate
        if start_sample <= edge_tolerance:
            keep_ms = leading_pause_ms
        elif end_sample >= data.size - edge_tolerance:
            keep_ms = trailing_pause_ms
        elif duration_ms >= long_threshold_ms:
            keep_ms = long_pause_ms
        else:
            keep_ms = medium_pause_ms
        keep_samples = max(0, int(round(sample_rate * keep_ms / 1000.0)))
        if keep_samples:
            parts.append(np.zeros(keep_samples, dtype=np.float32))
        cursor = end_sample
    parts.append(data[cursor:])
    return (np.concatenate(parts) if parts else data).astype(np.float32, copy=False)


def _run_ffmpeg(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        capture_output=capture,
        text=capture,
    )


def analyze_voice_mastering_profile(output_path: Path) -> dict[str, float | str]:
    """Measure the rendered voice and choose a restrained mastering profile."""
    audio, sr = sf.read(output_path, dtype="float32", always_2d=False)
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if data.size == 0:
        return {"tone": "neutral", "presence_gain_db": 1.0, "deesser_intensity": 0.10, "compressor_ratio": 1.8}
    # Cap spectral analysis to roughly one minute spread across the file.
    if data.size > sr * 60:
        indices = np.linspace(0, data.size - 1, sr * 60, dtype=np.int64)
        sample = data[indices]
    else:
        sample = data
    sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
    window = np.hanning(sample.size) if sample.size > 32 else np.ones(sample.size)
    spectrum = np.abs(np.fft.rfft(sample * window)) ** 2
    freqs = np.fft.rfftfreq(sample.size, d=1.0 / max(sr, 1))
    audible = (freqs >= 250) & (freqs <= min(9000, sr / 2 - 1))
    high = (freqs >= 4800) & (freqs <= min(9000, sr / 2 - 1))
    presence = (freqs >= 2200) & (freqs <= min(4200, sr / 2 - 1))
    audible_energy = float(np.sum(spectrum[audible]) + 1e-12)
    high_ratio = float(np.sum(spectrum[high]) / audible_energy)
    presence_ratio = float(np.sum(spectrum[presence]) / audible_energy)
    rms = float(np.sqrt(np.mean(np.square(sample, dtype=np.float64)) + 1e-12))
    peak = float(np.max(np.abs(sample)))
    crest_db = 20.0 * math.log10(max(peak, 1e-6) / max(rms, 1e-6))

    if high_ratio > 0.115 or presence_ratio > 0.35:
        tone = "bright"
        presence_gain = 0.45
        deesser = 0.16
    elif high_ratio < 0.035 and presence_ratio < 0.22:
        tone = "dark"
        presence_gain = 1.55
        deesser = 0.07
    else:
        tone = "neutral"
        presence_gain = 1.05
        deesser = 0.10
    ratio = 1.72 if crest_db > 16 else 1.50 if crest_db < 11 else 1.62
    return {
        "tone": tone,
        "presence_gain_db": round(presence_gain, 2),
        "deesser_intensity": round(deesser, 2),
        "compressor_ratio": round(ratio, 2),
        "high_frequency_ratio": round(high_ratio, 4),
        "presence_ratio": round(presence_ratio, 4),
        "crest_db": round(crest_db, 1),
    }


def master_professional_voice(output_path: Path) -> dict[str, float | str]:
    """Voice-aware spoken-word mastering for social-video playback.

    The source voice is measured before mastering. Bright voices receive less presence
    boost and more de-essing; darker voices receive a little more clarity. Compression
    remains gentle so 8–12 minute narration does not become fatiguing.
    """
    info = sf.info(output_path)
    profile = analyze_voice_mastering_profile(output_path)
    pre_path = output_path.with_name(output_path.stem + ".premaster.wav")
    out_path = output_path.with_name(output_path.stem + ".master.wav")
    pre_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)

    presence_gain = float(profile["presence_gain_db"])
    deesser = float(profile["deesser_intensity"])
    ratio = float(profile["compressor_ratio"])
    tone_chain = (
        "highpass=f=70,"
        "equalizer=f=180:t=q:w=0.8:g=-0.8,"
        f"equalizer=f=3200:t=q:w=1.0:g={presence_gain:.2f},"
        f"deesser=i={deesser:.2f}:m=0.30:f=0.55,"
        f"acompressor=threshold=0.125:ratio={ratio:.2f}:attack=24:release=220:makeup=1.06:knee=2.5:detection=rms"
    )
    try:
        _run_ffmpeg([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(output_path), "-filter:a", tone_chain,
            "-ar", str(info.samplerate), "-ac", "1", "-c:a", "pcm_s16le", str(pre_path),
        ])

        base = f"loudnorm=I={PRO_FINAL_TARGET_LUFS:.1f}:TP={PRO_FINAL_TRUE_PEAK_DBFS:.1f}:LRA={PRO_FINAL_LRA:.1f}"
        measured = _run_ffmpeg([
            "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info",
            "-i", str(pre_path), "-af", base + ":print_format=json", "-f", "null", "-",
        ], capture=True)
        stderr = measured.stderr or ""
        start = stderr.rfind("{")
        end = stderr.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("FFmpeg did not return loudness measurements")
        values = json.loads(stderr[start:end + 1])
        second = (
            base
            + f":measured_I={float(values['input_i']):.2f}"
            + f":measured_TP={float(values['input_tp']):.2f}"
            + f":measured_LRA={float(values['input_lra']):.2f}"
            + f":measured_thresh={float(values['input_thresh']):.2f}"
            + f":offset={float(values['target_offset']):.2f}"
            + ":linear=true:print_format=summary"
        )
        _run_ffmpeg([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(pre_path), "-filter:a", second,
            "-ar", str(info.samplerate), "-ac", "1", "-c:a", "pcm_s16le", str(out_path),
        ])
        out_path.replace(output_path)
        profile["target_lufs"] = PRO_FINAL_TARGET_LUFS
        profile["true_peak_dbfs"] = PRO_FINAL_TRUE_PEAK_DBFS
        return profile
    except (FileNotFoundError, subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Professional voice mastering requires a working FFmpeg audio filter stack.") from exc
    finally:
        pre_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


def create_video_master_48k_stereo(source_path: Path, destination_path: Path) -> None:
    """Create an editor/upload-friendly 48 kHz stereo spoken-word master.

    The voice-only master remains louder for preview. This companion has 1.5 dB more
    mix headroom for video editors, background ambience, and platform transcodes.
    """
    destination_path.unlink(missing_ok=True)
    try:
        _run_ffmpeg([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source_path), "-filter:a", "volume=-1.5dB,alimiter=limit=0.891:attack=5:release=50",
            "-ar", str(VIDEO_MASTER_SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_s16le", str(destination_path),
        ])
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        destination_path.unlink(missing_ok=True)
        raise RuntimeError("Creating the 48 kHz video master requires FFmpeg.") from exc

def adaptive_tempo_factor(
    *,
    current_wpm: float,
    target_wpm: float,
    min_wpm: float | None = None,
    max_wpm: float | None = None,
    min_factor: float = 0.84,
    max_factor: float = 1.04,
) -> float:
    """Return a pitch-preserving correction only outside the natural target band."""
    if current_wpm <= 0 or target_wpm <= 0:
        return 1.0
    if min_wpm is not None and max_wpm is not None and min_wpm <= current_wpm <= max_wpm:
        return 1.0
    return float(np.clip(target_wpm / current_wpm, min_factor, max_factor))

def apply_tempo_array(audio: np.ndarray, sample_rate: int, factor: float) -> np.ndarray:
    """Pitch-preserving FFmpeg atempo on a mono float32 waveform."""
    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    if data.size == 0 or abs(factor - 1.0) <= 0.002:
        return data.copy()
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
                "-filter:a", f"atempo={factor:.6f}",
                "-f", "f32le", "-ac", "1", "pipe:1",
            ],
            input=data.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Adaptive senior pacing requires FFmpeg atempo.") from exc
    out = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return out if out.size else data.copy()
