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


def master_professional_voice(output_path: Path, *, preserve_dynamics: bool = False) -> dict[str, float | str]:
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
    if preserve_dynamics:
        # Turbo's native emotion tags already create the useful performance movement.
        # Use mastering as a safety/clarity stage, not as a dynamics eraser.
        ratio = float(np.clip(ratio - 0.28, 1.24, 1.42))
        compressor = (
            f"acompressor=threshold=0.180:ratio={ratio:.2f}:attack=32:release=285:"
            "makeup=1.02:knee=2.2:detection=rms"
        )
    else:
        compressor = (
            f"acompressor=threshold=0.125:ratio={ratio:.2f}:attack=24:release=220:"
            "makeup=1.06:knee=2.5:detection=rms"
        )
    tone_chain = (
        "highpass=f=70,"
        "equalizer=f=180:t=q:w=0.8:g=-0.8,"
        f"equalizer=f=3200:t=q:w=1.0:g={presence_gain:.2f},"
        f"deesser=i={deesser:.2f}:m=0.30:f=0.55,"
        + compressor
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
        profile["preserve_dynamics"] = bool(preserve_dynamics)
        profile["compressor_ratio_used"] = round(float(ratio), 2)
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



def _measure_final_loudness(output_path: Path) -> dict[str, float]:
    """Measure final loudness without modifying the file."""
    base = f"loudnorm=I={PRO_FINAL_TARGET_LUFS:.1f}:TP={PRO_FINAL_TRUE_PEAK_DBFS:.1f}:LRA={PRO_FINAL_LRA:.1f}:print_format=json"
    try:
        measured = _run_ffmpeg([
            "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info",
            "-i", str(output_path), "-af", base, "-f", "null", "-",
        ], capture=True)
        stderr = measured.stderr or ""
        start, end = stderr.rfind("{"), stderr.rfind("}")
        if start < 0 or end <= start:
            return {}
        values = json.loads(stderr[start:end + 1])
        return {
            "integrated_lufs": float(values.get("input_i", -120.0)),
            "true_peak_dbfs": float(values.get("input_tp", -120.0)),
            "lra_lu": float(values.get("input_lra", 0.0)),
        }
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError):
        return {}


def analyze_prosody_quality(
    output_path: Path,
    emotion_summary: dict | None = None,
    *,
    avatar_minutes: float = 5.0,
) -> dict[str, float | int | str]:
    """Engineering heuristic for engagement movement in long-form spoken narration.

    This is not a perceptual MOS or a medical/industry standard. It intentionally
    complements Production QC by measuring the things that QC does not: front-loaded
    emotion spacing, final loudness range, short-term energy movement, and long flat
    stretches. It never rejects a job; it is an advisory score for tuning.
    """
    audio, sr = sf.read(output_path, dtype="float32", always_2d=False)
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = np.nan_to_num(data.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    duration = data.size / max(sr, 1)
    if data.size == 0 or duration <= 0:
        return {"score": 0.0, "rating": "unavailable", "duration_seconds": 0.0}

    def window_rms_db(start_s: float, end_s: float, window_s: float = 3.0, step_s: float = 1.5) -> list[float]:
        start = max(0, int(start_s * sr))
        end = min(data.size, int(end_s * sr))
        win = max(1, int(window_s * sr))
        step = max(1, int(step_s * sr))
        values: list[float] = []
        pos = start
        while pos + win <= end:
            frame = data[pos:pos + win]
            rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))
            values.append(20.0 * math.log10(max(rms, 1e-6)))
            pos += step
        return values

    loudness = _measure_final_loudness(output_path)
    lra = float(loudness.get("lra_lu", 0.0))
    avatar_seconds = min(duration, max(0.0, avatar_minutes * 60.0))
    first_values = window_rms_db(0.0, avatar_seconds)
    if first_values:
        active_threshold = max(-44.0, float(np.percentile(first_values, 80)) - 22.0)
        active = [v for v in first_values if v >= active_threshold]
        first_rms_range = float(np.percentile(active, 90) - np.percentile(active, 10)) if len(active) >= 4 else 0.0
    else:
        first_rms_range = 0.0

    # Longest nearly-flat active run in non-overlapping 5-second windows.
    flat_values = window_rms_db(0.0, avatar_seconds, window_s=5.0, step_s=5.0)
    longest_flat = 0.0
    current_flat = 0.0
    previous: float | None = None
    for value in flat_values:
        if value < -44.0:
            current_flat = 0.0
            previous = None
            continue
        if previous is not None and abs(value - previous) <= 0.45:
            current_flat += 5.0
        else:
            current_flat = 5.0
        longest_flat = max(longest_flat, current_flat)
        previous = value

    auto_emotion_enabled = emotion_summary is not None
    summary = emotion_summary or {}
    placements = list(summary.get("placements") or [])
    total_words = max(1, int(summary.get("total_words") or 1))
    cue_times: list[float] = []
    for item in placements:
        try:
            position = max(0, int(item.get("word_position", 0)))
        except (TypeError, ValueError):
            continue
        estimated_time = duration * min(position / total_words, 1.0)
        if estimated_time <= avatar_seconds + 1e-6:
            cue_times.append(estimated_time)
    cue_times = sorted(set(round(value, 3) for value in cue_times))
    avatar_tag_count = len(cue_times)
    if auto_emotion_enabled:
        avatar_target = max(1, int(summary.get("avatar_target_count") or min(10, max(2, round(total_words / 120)))))
        anchors = [0.0] + cue_times + [avatar_seconds]
        max_gap = max((b - a for a, b in zip(anchors, anchors[1:])), default=avatar_seconds)
        emotion_points = 30.0 * min(avatar_tag_count / max(avatar_target, 1), 1.0)
        if max_gap <= 45:
            gap_points = 20.0
        elif max_gap <= 60:
            gap_points = 17.0
        elif max_gap <= 75:
            gap_points = 13.0
        elif max_gap <= 100:
            gap_points = 8.0
        else:
            gap_points = 3.0
    else:
        # Auto Emotion is deliberately optional in v1.5.7. Do not punish a creator
        # for turning it off; score the acoustic prosody/mastering on its own.
        avatar_target = 0
        max_gap = 0.0
        emotion_points = 0.0
        gap_points = 0.0
    if 4.0 <= lra <= 7.5:
        lra_points = 25.0
    elif lra > 7.5:
        lra_points = max(17.0, 25.0 - (lra - 7.5) * 1.8)
    else:
        lra_points = 25.0 * min(max(lra, 0.0) / 4.0, 1.0)
    movement_points = 15.0 * min(first_rms_range / 4.0, 1.0)
    if longest_flat <= 35:
        flat_points = 10.0
    elif longest_flat <= 50:
        flat_points = 8.0
    elif longest_flat <= 70:
        flat_points = 5.0
    else:
        flat_points = 2.0
    if auto_emotion_enabled:
        score = float(np.clip(emotion_points + gap_points + lra_points + movement_points + flat_points, 0.0, 100.0))
    else:
        score = float(np.clip(
            (lra_points / 25.0) * 45.0
            + (movement_points / 15.0) * 35.0
            + (flat_points / 10.0) * 20.0,
            0.0, 100.0,
        ))
    rating = "excellent" if score >= 85 else "strong" if score >= 75 else "good" if score >= 65 else "needs more movement"
    return {
        "score": round(score, 1),
        "rating": rating,
        "duration_seconds": round(duration, 2),
        "avatar_seconds": round(avatar_seconds, 2),
        "auto_emotion_enabled": auto_emotion_enabled,
        "avatar_tags": avatar_tag_count,
        "avatar_target_tags": avatar_target,
        "max_avatar_emotion_gap_seconds": round(max_gap, 1),
        "lra_lu": round(lra, 2),
        "first5_rms_range_db": round(first_rms_range, 2),
        "longest_flat_stretch_seconds": round(longest_flat, 1),
        "integrated_lufs": round(float(loudness.get("integrated_lufs", -120.0)), 2),
        "true_peak_dbfs": round(float(loudness.get("true_peak_dbfs", -120.0)), 2),
        "note": "Advisory engineering score; it does not reject audio.",
    }
