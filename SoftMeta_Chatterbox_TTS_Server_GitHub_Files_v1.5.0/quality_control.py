from __future__ import annotations

import math
import os
import re
import statistics
import tempfile
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from emotion_director import strip_turbo_tags

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)


def normalize_words(text: str) -> list[str]:
    text = strip_turbo_tags(text).lower()
    text = text.replace("’", "'")
    return _WORD.findall(text)


def word_error_rate(expected: str, actual: str) -> float:
    ref = normalize_words(expected)
    hyp = normalize_words(actual)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rword in enumerate(ref, start=1):
        current = [i]
        for j, hword in enumerate(hyp, start=1):
            current.append(min(
                current[-1] + 1,
                prev[j] + 1,
                prev[j - 1] + (rword != hword),
            ))
        prev = current
    return prev[-1] / len(ref)


def acoustic_metrics(audio: np.ndarray, sample_rate: int, expected_words: int) -> dict[str, float]:
    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return {"rms_dbfs": -120.0, "peak_dbfs": -120.0, "silence_ratio": 1.0, "wpm": 0.0}
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64)) + 1e-12))
    peak = float(np.max(np.abs(data)))
    duration = data.size / max(sample_rate, 1)
    frame = max(1, int(sample_rate * 0.02))
    count = max(1, data.size // frame)
    frames = data[:count * frame].reshape(count, frame)
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    db = 20 * np.log10(np.maximum(frame_rms, 1e-6))
    speech_ref = float(np.percentile(db, 80))
    threshold = float(np.clip(speech_ref - 28.0, -54.0, -40.0))
    silence = float(np.mean(db <= threshold))
    return {
        "rms_dbfs": 20 * math.log10(max(rms, 1e-6)),
        "peak_dbfs": 20 * math.log10(max(peak, 1e-6)),
        "silence_ratio": silence,
        "wpm": expected_words * 60.0 / duration if duration > 0 and expected_words else 0.0,
    }


def _has_repetition_loop(text: str) -> bool:
    words = normalize_words(text)
    if len(words) < 12:
        return False
    grams: dict[tuple[str, ...], int] = {}
    for size in (3, 4, 5):
        grams.clear()
        for i in range(len(words) - size + 1):
            gram = tuple(words[i:i + size])
            grams[gram] = grams.get(gram, 0) + 1
        if any(count >= 3 for count in grams.values()):
            return True
    return False


@dataclass(slots=True)
class ChunkQualityReport:
    passed: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    transcript: str = ""
    wer: float | None = None
    speaker_similarity: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    words: list[dict[str, Any]] = field(default_factory=list)
    asr_available: bool = False
    speaker_check_available: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(float(self.score), 1),
            "reasons": list(self.reasons),
            "transcript": self.transcript,
            "wer": None if self.wer is None else round(float(self.wer), 3),
            "speaker_similarity": None if self.speaker_similarity is None else round(float(self.speaker_similarity), 3),
            "metrics": {k: round(float(v), 3) for k, v in self.metrics.items()},
            "asr_available": self.asr_available,
            "speaker_check_available": self.speaker_check_available,
        }


class QualityController:
    """Lazy production QC used by both Chatterbox Original and Turbo.

    ASR and speaker verification are optional at import time so the server can start
    even before their models are downloaded. Colab installs both dependencies. If a
    model download or CUDA backend fails, QC keeps the acoustic checks active rather
    than failing the user's entire audio job.
    """

    def __init__(self) -> None:
        self.asr_model_name = os.getenv("SOFTMETA_ASR_MODEL", "small.en")
        self._asr: Any | None = None
        self._asr_failed = False
        self._speaker: Any | None = None
        self._speaker_failed = False
        self._reference_cache: dict[tuple[str, float], Path] = {}

    def _load_asr(self) -> Any | None:
        if self._asr is not None:
            return self._asr
        if self._asr_failed:
            return None
        try:
            from faster_whisper import WhisperModel
            device = "cuda" if os.getenv("SOFTMETA_DEVICE", "auto") == "cuda" else "auto"
            compute = "float16" if device == "cuda" else "int8"
            try:
                self._asr = WhisperModel(self.asr_model_name, device=device, compute_type=compute)
            except Exception:
                self._asr = WhisperModel(self.asr_model_name, device="cpu", compute_type="int8")
            return self._asr
        except Exception:
            self._asr_failed = True
            return None

    def _load_speaker(self) -> Any | None:
        if self._speaker is not None:
            return self._speaker
        if self._speaker_failed:
            return None
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
            cache = os.getenv("SOFTMETA_SPEAKER_CACHE", "/content/hf_home/softmeta-speaker")
            self._speaker = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=cache,
                run_opts={"device": "cpu"},
            )
            return self._speaker
        except Exception:
            self._speaker_failed = True
            return None

    @staticmethod
    def _temp_wav(audio: np.ndarray, sample_rate: int) -> Path:
        handle = tempfile.NamedTemporaryFile(prefix="softmeta_qc_", suffix=".wav", delete=False)
        handle.close()
        path = Path(handle.name)
        sf.write(path, np.asarray(audio, dtype=np.float32).reshape(-1), sample_rate, subtype="PCM_16")
        return path

    def transcribe(self, audio: np.ndarray, sample_rate: int, expected_text: str) -> tuple[str, list[dict[str, Any]], bool]:
        model = self._load_asr()
        if model is None:
            return "", [], False
        path = self._temp_wav(audio, sample_rate)
        try:
            segments, _ = model.transcribe(
                str(path), language="en", beam_size=3, temperature=0.0,
                word_timestamps=True, vad_filter=True,
                initial_prompt=strip_turbo_tags(expected_text)[:800],
                hotwords=strip_turbo_tags(expected_text)[:400],
                condition_on_previous_text=False,
            )
            transcript_parts: list[str] = []
            words: list[dict[str, Any]] = []
            for segment in segments:
                transcript_parts.append(segment.text.strip())
                for word in segment.words or []:
                    words.append({"start": float(word.start), "end": float(word.end), "word": str(word.word)})
            return " ".join(part for part in transcript_parts if part).strip(), words, True
        except Exception:
            return "", [], False
        finally:
            path.unlink(missing_ok=True)

    def _speaker_wav(self, audio: np.ndarray, sample_rate: int) -> Path:
        """Normalize generated speech to the ECAPA model's 16 kHz mono domain."""
        source = self._temp_wav(audio, sample_rate)
        handle = tempfile.NamedTemporaryFile(prefix="softmeta_spk_", suffix=".wav", delete=False)
        handle.close()
        destination = Path(handle.name)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(destination),
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return destination
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            source.unlink(missing_ok=True)

    def _reference_wav(self, reference_path: Path) -> Path:
        key = (str(reference_path.resolve()), float(reference_path.stat().st_mtime))
        cached = self._reference_cache.get(key)
        if cached is not None and cached.exists():
            return cached
        handle = tempfile.NamedTemporaryFile(prefix="softmeta_ref_", suffix=".wav", delete=False)
        handle.close()
        destination = Path(handle.name)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(reference_path),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(destination),
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        for old_key, old_path in list(self._reference_cache.items()):
            if old_key[0] == key[0] and old_key != key:
                old_path.unlink(missing_ok=True)
                self._reference_cache.pop(old_key, None)
        self._reference_cache[key] = destination
        return destination

    def speaker_similarity(self, reference_path: Path | None, audio: np.ndarray, sample_rate: int) -> tuple[float | None, bool]:
        if reference_path is None:
            return None, False
        model = self._load_speaker()
        if model is None:
            return None, False
        generated = self._speaker_wav(audio, sample_rate)
        try:
            normalized_reference = self._reference_wav(reference_path)
            score, _prediction = model.verify_files(str(normalized_reference), str(generated))
            value = float(score.detach().cpu().reshape(-1)[0]) if hasattr(score, "detach") else float(score)
            return value, True
        except Exception:
            return None, False
        finally:
            generated.unlink(missing_ok=True)

    def evaluate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        expected_text: str,
        reference_path: Path | None,
        speaker_history: list[float] | None = None,
    ) -> ChunkQualityReport:
        expected_words = len(normalize_words(expected_text))
        metrics = acoustic_metrics(audio, sample_rate, expected_words)
        reasons: list[str] = []
        severe = False

        if metrics["peak_dbfs"] > -0.03:
            reasons.append("near-clipping peak"); severe = True
        if metrics["rms_dbfs"] < -42:
            reasons.append("very low speech level"); severe = True
        if metrics["silence_ratio"] > 0.55:
            reasons.append("excessive silence"); severe = True
        if metrics["wpm"] and (metrics["wpm"] < 85 or metrics["wpm"] > 220):
            reasons.append("abnormal speaking rate"); severe = True

        transcript, words, asr_available = self.transcribe(audio, sample_rate, expected_text)
        wer: float | None = None
        if asr_available and expected_words >= 5:
            wer = word_error_rate(expected_text, transcript)
            ratio = len(normalize_words(transcript)) / max(expected_words, 1)
            if wer > 0.30:
                reasons.append(f"high ASR mismatch ({wer:.0%})"); severe = True
            elif wer > 0.20:
                reasons.append(f"moderate ASR mismatch ({wer:.0%})")
            if ratio < 0.72:
                reasons.append("possible missing words"); severe = True
            elif ratio > 1.30 or _has_repetition_loop(transcript):
                reasons.append("possible repeated or hallucinated words"); severe = True

        speaker_similarity, speaker_available = self.speaker_similarity(reference_path, audio, sample_rate)
        history = speaker_history or []
        if speaker_similarity is not None:
            if speaker_similarity < 0.02:
                reasons.append("very low reference-speaker similarity"); severe = True
            elif len(history) >= 2:
                baseline = statistics.median(history[-5:])
                if speaker_similarity < baseline - 0.16:
                    reasons.append("speaker identity drift"); severe = True
                elif speaker_similarity < baseline - 0.10:
                    reasons.append("possible speaker identity drift")

        score = 100.0
        if wer is not None:
            score -= min(52.0, wer * 135.0)
        if metrics["silence_ratio"] > 0.30:
            score -= min(18.0, (metrics["silence_ratio"] - 0.30) * 45.0)
        if speaker_similarity is not None and history:
            baseline = statistics.median(history[-5:])
            score -= max(0.0, baseline - speaker_similarity) * 80.0
        if severe:
            score -= 12.0
        score = max(0.0, min(100.0, score))
        return ChunkQualityReport(
            passed=not severe,
            score=score,
            reasons=reasons,
            transcript=transcript,
            wer=wer,
            speaker_similarity=speaker_similarity,
            metrics=metrics,
            words=words,
            asr_available=asr_available,
            speaker_check_available=speaker_available,
        )


def summarize_quality(reports: list[ChunkQualityReport], retries: int) -> dict[str, Any]:
    wers = [r.wer for r in reports if r.wer is not None]
    sims = [r.speaker_similarity for r in reports if r.speaker_similarity is not None]
    return {
        "checked_chunks": len(reports),
        "passed_chunks": sum(1 for r in reports if r.passed),
        "retries": retries,
        "average_score": round(float(np.mean([r.score for r in reports])), 1) if reports else None,
        "average_wer": round(float(np.mean(wers)), 3) if wers else None,
        "minimum_speaker_similarity": round(float(min(sims)), 3) if sims else None,
        "asr_verified": any(r.asr_available for r in reports),
        "speaker_verified": any(r.speaker_check_available for r in reports),
        "warnings": [reason for r in reports for reason in r.reasons],
    }
