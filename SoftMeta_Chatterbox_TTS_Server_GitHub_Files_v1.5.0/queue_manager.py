from __future__ import annotations

import asyncio
import json
import logging
import math
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from engine import EngineService
from emotion_director import analyze_serious_senior_advisor, contains_turbo_tag, is_heading, strip_turbo_tags
from models import AudioJobCreate
from storage import Storage
from utils import (
    count_words,
    estimated_audio_seconds,
    safe_filename,
    split_text,
)
from speech_pipeline import build_long_form_segments, prepare_senior_clear_speech_text
from pronunciation_engine import prepare_pronunciation_text
from quality_control import QualityController, summarize_quality
from captions import aligned_expected_words, fallback_words, write_caption_files
from reference_quality import analyze_reference_voice
from professional_audio import (
    adaptive_tempo_factor,
    apply_tempo_array,
    master_professional_voice,
    shape_professional_pauses,
    create_video_master_48k_stereo,
)

logger = logging.getLogger(__name__)

TERMINAL_STATES = {"completed", "failed", "cancelled", "interrupted"}
ACTIVE_STATES = {"queued", "running"}

MOTIVATIONAL_TURBO_PROFILE: dict[str, Any] = {
    # Turbo has no CFG/exaggeration control. Keep sampling deliberately tighter
    # than the generic defaults so independent long-form chunks are less likely
    # to jump into an excited/angry prosody or garble a phrase.
    "temperature": 0.60,
    "exaggeration": 0.0,
    "cfg_weight": 0.0,
    "repetition_penalty": 1.18,
    "min_p": 0.0,
    "top_p": 0.85,
    "top_k": 600,
    # Age-aware target bands own the final senior-advisor pace. 85 words
    # is a stability/speed compromise: it limits within-chunk prosody drift while
    # retaining most of Turbo's long-form speed advantage on an L4.
    "speed_factor": 1.0,
    "chunk_words": 85,
    # Keep chunk joins to a human breath-sized pause. The Excessive Silence Guard
    # below handles any multi-second silence hallucinated *inside* a Turbo chunk.
    "inter_chunk_pause_ms": 220,
    # Turbo is English-only. This locks the server path to English text processing;
    # cloned accent identity can still be inherited from the reference recording.
    "language": "en",
}

MOTIVATIONAL_TURBO_RETRY_PROFILE: dict[str, Any] = {
    "temperature": 0.50,
    "top_p": 0.80,
    "top_k": 400,
}

TURBO_CHUNK_TARGET_RMS_DBFS = -17.0
TURBO_CHUNK_PEAK_CEILING_DBFS = -1.5
TURBO_FINAL_TARGET_LUFS = -12.5
TURBO_FINAL_TRUE_PEAK_DBFS = -0.8

PROFESSIONAL_MODELS = {"chatterbox", "chatterbox-turbo"}

MOTIVATIONAL_ORIGINAL_PROFILE: dict[str, Any] = {
    "temperature": 0.72,
    "exaggeration": 0.58,
    "cfg_weight": 0.35,
    "repetition_penalty": 1.20,
    "min_p": 0.05,
    "top_p": 1.0,
    "top_k": 1000,
    "speed_factor": 1.0,
    "chunk_words": 85,
    "inter_chunk_pause_ms": 220,
    "language": "en",
}

MOTIVATIONAL_ORIGINAL_RETRY_PROFILE: dict[str, Any] = {
    "temperature": 0.60,
    "exaggeration": 0.52,
    "cfg_weight": 0.32,
}

ORIGINAL_EMOTION_PROFILE: dict[str, dict[str, float]] = {
    "narration": {"temperature": 0.68, "exaggeration": 0.54, "cfg_weight": 0.36},
    "happy": {"temperature": 0.72, "exaggeration": 0.58, "cfg_weight": 0.35},
    "surprised": {"temperature": 0.68, "exaggeration": 0.60, "cfg_weight": 0.34},
    "dramatic": {"temperature": 0.64, "exaggeration": 0.62, "cfg_weight": 0.32},
}

class QueueManager:
    """Single-worker GPU queue for up to five prepared audio jobs.

    The browser may navigate between tabs, minimise the progress dialog or reload;
    generation remains owned by this server-side worker.
    """

    def __init__(self, engine: EngineService, storage: Storage) -> None:
        self.engine = engine
        self.storage = storage
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.jobs: dict[str, dict[str, Any]] = self.storage.load_jobs()
        self.worker_task: asyncio.Task | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="softmeta-tts")
        self._waiters: dict[str, asyncio.Event] = {}
        self._last_persist = 0.0
        self.quality = QualityController()

        for job in self.jobs.values():
            if job.get("status") in ACTIVE_STATES:
                job["status"] = "interrupted"
                job["stage"] = "Runtime restarted before this job completed."
                job["completed_at"] = time.time()
        self._persist(force=True)

    def _persist(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_persist >= 1.5:
            self.storage.save_jobs(self.jobs)
            self._last_persist = now

    async def start(self) -> None:
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker(), name="softmeta-audio-queue")

    async def stop(self) -> None:
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=False, cancel_futures=True)
        self._persist(force=True)

    def _queue_positions(self) -> dict[str, int]:
        queued = [job for job in self.jobs.values() if job.get("status") == "queued"]
        queued.sort(key=lambda item: item.get("created_at", 0.0))
        return {job["id"]: index for index, job in enumerate(queued, start=1)}

    def public_job(self, job: dict[str, Any]) -> dict[str, Any]:
        data = dict(job)
        data.pop("cancel_requested", None)
        data.pop("generation_text", None)
        data["queue_position"] = self._queue_positions().get(job["id"])
        if data.get("started_at"):
            end = data.get("completed_at") or time.time()
            data["elapsed_seconds"] = max(0, round(end - data["started_at"]))
        else:
            data["elapsed_seconds"] = 0
        return data

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs = sorted(self.jobs.values(), key=lambda item: item.get("created_at", 0.0))
        return [self.public_job(job) for job in jobs]

    def get_raw(self, job_id: str) -> dict[str, Any]:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    def get(self, job_id: str) -> dict[str, Any]:
        return self.public_job(self.get_raw(job_id))

    def has_active_jobs(self) -> bool:
        return any(job.get("status") in ACTIVE_STATES for job in self.jobs.values())

    @staticmethod
    def _effective_options(request: AudioJobCreate) -> dict[str, Any]:
        options = request.options.model_dump()
        if request.preset == "Motivational Speech":
            if options.get("model") == "chatterbox-turbo":
                # Turbo lacks Original's native CFG/exaggeration controls, so its
                # stable senior-advisor sampling profile is backend-enforced.
                options.update(MOTIVATIONAL_TURBO_PROFILE)
            elif options.get("model") == "chatterbox":
                # Original keeps the creator's native CFG/exaggeration/sampling
                # controls. Only structural professional defaults are shared here;
                # semantic direction and failed-chunk retry are applied per chunk.
                options.update({
                    "chunk_words": MOTIVATIONAL_ORIGINAL_PROFILE["chunk_words"],
                    "inter_chunk_pause_ms": MOTIVATIONAL_ORIGINAL_PROFILE["inter_chunk_pause_ms"],
                    "language": MOTIVATIONAL_ORIGINAL_PROFILE["language"],
                })
        return options

    @staticmethod
    def _original_chunk_direction(chunk: str, options: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        directed = dict(options)
        is_quality_retry = bool(directed.pop("_quality_retry", False))
        tag_match = __import__("re").search(r"\[(happy|surprised|dramatic|narration)\]", chunk, flags=__import__("re").IGNORECASE)
        if tag_match:
            directed.update(ORIGINAL_EMOTION_PROFILE.get(tag_match.group(1).lower(), {}))
        # A failed Original chunk must genuinely become more conservative on retry.
        # Apply the safety profile *after* the semantic direction so emotion cannot
        # accidentally restore the looser first-pass temperature/exaggeration.
        if is_quality_retry:
            directed.update(MOTIVATIONAL_ORIGINAL_RETRY_PROFILE)
        return strip_turbo_tags(chunk).strip(), directed

    @staticmethod
    def _apply_final_tempo(output_path: Path, speed_factor: float) -> None:
        """Apply pitch-preserving playback tempo once after all chunks are joined."""
        if abs(speed_factor - 1.0) <= 0.001:
            return
        temp_path = output_path.with_name(output_path.stem + ".tempo.wav")
        temp_path.unlink(missing_ok=True)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(output_path),
                    "-filter:a", f"atempo={speed_factor:.6f}",
                    "-c:a", "pcm_s16le",
                    str(temp_path),
                ],
                check=True,
            )
            temp_path.replace(output_path)
        except (FileNotFoundError, subprocess.CalledProcessError):
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Pitch-preserving speed adjustment requires FFmpeg. "
                "Install ffmpeg and try again."
            )


    @staticmethod
    def _compact_excessive_silence(
        audio: np.ndarray,
        sample_rate: int,
        *,
        min_silence_ms: int = 900,
        internal_pause_ms: int = 340,
        leading_pause_ms: int = 80,
        trailing_pause_ms: int = 120,
    ) -> np.ndarray:
        """Collapse only abnormally long dead-air regions in Turbo output.

        Chatterbox Turbo can occasionally emit several seconds of near-silence after
        punctuation or an expressive-token boundary. That silence is inside the
        generated waveform, so changing the server's inter-chunk pause cannot fix it.
        Short natural breaths remain untouched; only near-silence lasting at least
        ``min_silence_ms`` is replaced with a clean human-sized pause.
        """
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

        # Adaptive threshold: far enough below speech to preserve quiet syllables and
        # normal breaths, but high enough to recognize low-level generated room hiss.
        speech_reference = float(np.percentile(db, 80))
        silence_threshold = float(np.clip(speech_reference - 27.0, -50.0, -38.0))
        silent = db <= silence_threshold
        min_frames = max(1, int(math.ceil(min_silence_ms / 20.0)))

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
            is_leading = start_sample <= edge_tolerance
            is_trailing = end_sample >= data.size - edge_tolerance
            if is_leading:
                keep_ms = leading_pause_ms
            elif is_trailing:
                keep_ms = trailing_pause_ms
            else:
                keep_ms = internal_pause_ms
            keep_samples = max(0, int(round(sample_rate * keep_ms / 1000.0)))
            if keep_samples:
                # Use true digital silence so generated hiss/box noise in an abnormal
                # pause is not carried into the final narration.
                parts.append(np.zeros(keep_samples, dtype=np.float32))
            cursor = end_sample
        parts.append(data[cursor:])
        compacted = np.concatenate(parts) if parts else data
        return compacted.astype(np.float32, copy=False)

    @staticmethod
    def _audio_metrics(audio: np.ndarray, sample_rate: int, words: int) -> dict[str, float]:
        data = np.asarray(audio, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return {"rms_dbfs": -120.0, "peak_dbfs": -120.0, "seconds_per_word": 0.0}
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64)) + 1e-12))
        peak = float(np.max(np.abs(data)))
        duration = data.size / max(sample_rate, 1)
        return {
            "rms_dbfs": 20.0 * math.log10(max(rms, 1e-6)),
            "peak_dbfs": 20.0 * math.log10(max(peak, 1e-6)),
            "seconds_per_word": duration / max(words, 1),
        }

    @staticmethod
    def _level_turbo_chunk(audio: np.ndarray) -> np.ndarray:
        """Level each Turbo chunk before concatenation.

        Final-file LUFS alone can leave most of a narration quiet when one or two
        generated chunks are unexpectedly loud. Per-chunk RMS matching removes that
        imbalance first, then a conservative peak ceiling prevents clipping.
        """
        data = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
        if data.size == 0:
            return data
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64)) + 1e-12))
        if rms > 1e-6:
            current_db = 20.0 * math.log10(rms)
            gain_db = float(np.clip(TURBO_CHUNK_TARGET_RMS_DBFS - current_db, -12.0, 24.0))
            data *= 10.0 ** (gain_db / 20.0)
        peak = float(np.max(np.abs(data)))
        ceiling = 10.0 ** (TURBO_CHUNK_PEAK_CEILING_DBFS / 20.0)
        if peak > ceiling and peak > 0.0:
            data *= ceiling / peak
        return data

    @staticmethod
    def _turbo_chunk_is_unstable(
        metrics: dict[str, float],
        history: list[dict[str, float]],
        *,
        intentional_emotion: bool = False,
    ) -> bool:
        """Detect obvious Turbo chunk drift without adding an ASR dependency."""
        rms_db = metrics["rms_dbfs"]
        spw = metrics["seconds_per_word"]
        if not math.isfinite(rms_db) or not math.isfinite(spw) or spw <= 0.0:
            return True
        if rms_db > -6.0 or rms_db < -48.0 or spw < 0.16 or spw > 0.90:
            return True
        if len(history) < 2:
            return False
        # Native Turbo emotion tokens intentionally alter prosody. For those chunks,
        # keep only the absolute sanity guard above; per-chunk level matching still
        # prevents the expressive moment from becoming a volume spike.
        if intentional_emotion:
            return False
        recent = history[-5:]
        median_rms = float(np.median([item["rms_dbfs"] for item in recent]))
        median_spw = float(np.median([item["seconds_per_word"] for item in recent]))
        # Sudden loudness or speaking-rate jumps are strong proxies for the
        # angry/fast/garbled chunk failures observed in long Turbo narration.
        if rms_db > median_rms + 5.0 or rms_db < median_rms - 9.0:
            return True
        if median_spw > 0.0 and (spw < median_spw * 0.62 or spw > median_spw * 1.65):
            return True
        return False

    @staticmethod
    def _chunk_distance_from_history(
        metrics: dict[str, float],
        history: list[dict[str, float]],
    ) -> float:
        if not history:
            return 0.0
        recent = history[-5:]
        median_rms = float(np.median([item["rms_dbfs"] for item in recent]))
        median_spw = float(np.median([item["seconds_per_word"] for item in recent]))
        rms_distance = abs(metrics["rms_dbfs"] - median_rms) / 6.0
        speed_distance = abs(metrics["seconds_per_word"] - median_spw) / max(median_spw, 0.1)
        return rms_distance + speed_distance

    @staticmethod
    def _normalize_turbo_loudness(output_path: Path) -> None:
        """Two-pass EBU R128 mastering for predictable Turbo output loudness.

        The first pass measures the complete narration. The second pass uses those
        measurements for linear loudness correction, which is much more reliable
        than the previous one-pass filter when long files contain a few unusually
        loud chunks.
        """
        info = sf.info(output_path)
        temp_path = output_path.with_name(output_path.stem + ".loud.wav")
        temp_path.unlink(missing_ok=True)
        base_filter = (
            f"loudnorm=I={TURBO_FINAL_TARGET_LUFS:.1f}:"
            f"TP={TURBO_FINAL_TRUE_PEAK_DBFS:.1f}:LRA=5.0"
        )
        try:
            measure = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info",
                    "-i", str(output_path),
                    "-af", base_filter + ":print_format=json",
                    "-f", "null", "-",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stderr = measure.stderr
            json_start = stderr.rfind("{")
            json_end = stderr.rfind("}")
            if json_start < 0 or json_end <= json_start:
                raise RuntimeError("FFmpeg did not return Turbo loudness measurements.")
            measured = json.loads(stderr[json_start:json_end + 1])
            second_filter = (
                base_filter
                + f":measured_I={float(measured['input_i']):.2f}"
                + f":measured_TP={float(measured['input_tp']):.2f}"
                + f":measured_LRA={float(measured['input_lra']):.2f}"
                + f":measured_thresh={float(measured['input_thresh']):.2f}"
                + f":offset={float(measured['target_offset']):.2f}"
                + ":linear=true:print_format=summary"
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(output_path),
                    "-filter:a", second_filter,
                    "-ar", str(info.samplerate),
                    "-ac", "1",
                    "-c:a", "pcm_s16le",
                    str(temp_path),
                ],
                check=True,
            )
            temp_path.replace(output_path)
        except (FileNotFoundError, subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as error:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Turbo loudness normalisation requires a working FFmpeg loudnorm filter. "
                "Install FFmpeg and try again."
            ) from error

    async def create(self, request: AudioJobCreate, enqueue: bool = True) -> dict[str, Any]:
        total_words = count_words(request.text)
        if total_words == 0:
            raise ValueError("The script is empty.")
        job_id = uuid.uuid4().hex
        now = time.time()
        effective_options = self._effective_options(request)
        generation_text = request.text
        emotion_summary: dict[str, Any] | None = None
        if (
            request.preset == "Motivational Speech"
            and effective_options.get("model") in PROFESSIONAL_MODELS
        ):
            emotion_analysis = analyze_serious_senior_advisor(request.text)
            pronounced = prepare_pronunciation_text(emotion_analysis.tagged_text)
            generation_text = prepare_senior_clear_speech_text(pronounced)
            emotion_summary = emotion_analysis.public_summary(include_text=False)
        job = {
            "id": job_id,
            "preset": request.preset,
            "audio_number": request.audio_number,
            "title": request.title.strip(),
            "text": request.text,
            "generation_text": generation_text,
            "emotion_summary": emotion_summary,
            "voice_mode": request.voice_mode,
            "voice_filename": request.voice_filename,
            "options": effective_options,
            "status": "queued" if enqueue else "draft",
            "stage": "Waiting in queue" if enqueue else "Draft",
            "total_words": total_words,
            "completed_words": 0,
            "display_words": 0,
            "percent": 0.0,
            "remaining_percent": 100.0,
            "eta_seconds": None,
            "estimated_audio_seconds": round(
                estimated_audio_seconds(total_words, float(effective_options["speed_factor"])), 1
            ),
            "actual_audio_seconds": None,
            "output_filename": None,
            "video_master_filename": None,
            "srt_filename": None,
            "vtt_filename": None,
            "quality_summary": None,
            "reference_quality": None,
            "mastering_profile": None,
            "error": None,
            "cancel_requested": False,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
        }
        self.jobs[job_id] = job
        self._waiters[job_id] = asyncio.Event()
        self._persist(force=True)
        if enqueue:
            await self.queue.put(job_id)
        return self.public_job(job)

    async def create_many(self, requests: list[AudioJobCreate]) -> list[dict[str, Any]]:
        created: list[dict[str, Any]] = []
        for request in requests:
            created.append(await self.create(request, enqueue=True))
        return created

    async def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get_raw(job_id)
        if job["status"] in TERMINAL_STATES:
            return self.public_job(job)
        job["cancel_requested"] = True
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["stage"] = "Cancelled before generation"
            job["completed_at"] = time.time()
            self._signal(job_id)
        else:
            job["stage"] = "Cancellation requested; waiting for the current model pass"
        self._persist(force=True)
        return self.public_job(job)

    async def delete(self, job_id: str, *, delete_file: bool = True) -> None:
        job = self.get_raw(job_id)
        if job.get("status") in ACTIVE_STATES:
            raise RuntimeError("Active jobs cannot be removed. Wait for generation or cancel the job first.")
        filename = job.get("output_filename")
        if delete_file and filename:
            self.storage.delete_output_artifacts(filename)
        self.jobs.pop(job_id, None)
        self._waiters.pop(job_id, None)
        self._persist(force=True)

    async def clear(self, *, delete_files: bool = True) -> None:
        if self.has_active_jobs():
            raise RuntimeError("Wait for all queued audio jobs to finish before using Remove All.")
        if delete_files:
            for job in self.jobs.values():
                filename = job.get("output_filename")
                if filename:
                    self.storage.delete_output_artifacts(filename)
            self.storage.clear_outputs()
        self.jobs.clear()
        self._waiters.clear()
        self._persist(force=True)

    async def wait(self, job_id: str, timeout: float | None = None) -> dict[str, Any]:
        job = self.get_raw(job_id)
        if job["status"] in TERMINAL_STATES:
            return self.public_job(job)
        event = self._waiters.setdefault(job_id, asyncio.Event())
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return self.get(job_id)

    def _signal(self, job_id: str) -> None:
        self._waiters.setdefault(job_id, asyncio.Event()).set()

    async def _worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                job = self.jobs.get(job_id)
                if not job or job["status"] == "cancelled":
                    continue
                await self._process(job)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                job = self.jobs.get(job_id)
                if job:
                    job["status"] = "failed"
                    job["stage"] = "Generation failed"
                    job["error"] = f"{type(error).__name__}: {error}"
                    job["completed_at"] = time.time()
                    job["eta_seconds"] = 0
                    self._persist(force=True)
                    self._signal(job_id)
            finally:
                self.queue.task_done()

    def _resolve_voice(self, job: dict[str, Any]) -> Path | None:
        if job["voice_mode"] == "default":
            return None
        return self.storage.voice_path(job["voice_mode"], job["voice_filename"])

    @staticmethod
    def _update_percent(job: dict[str, Any], cap: float = 95.0) -> None:
        total = max(int(job["total_words"]), 1)
        percent = min(cap, (float(job["display_words"]) / total) * 95.0)
        job["percent"] = round(percent, 1)
        job["remaining_percent"] = round(max(0.0, 100.0 - percent), 1)

    async def _smooth_words(
        self,
        job: dict[str, Any],
        base_words: int,
        active_words: int,
        expected_seconds: float,
    ) -> None:
        started = time.monotonic()
        try:
            while job["status"] == "running":
                elapsed = time.monotonic() - started
                fraction = min(0.92, elapsed / max(expected_seconds, 1.0))
                job["display_words"] = min(
                    max(job["total_words"] - 1, 0),
                    base_words + int(active_words * fraction),
                )
                self._update_percent(job)
                job["eta_seconds"] = max(1, round(expected_seconds - elapsed))
                self._persist()
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    async def _process(self, job: dict[str, Any]) -> None:
        job["status"] = "running"
        job["stage"] = "Preparing text and voice"
        job["started_at"] = time.time()
        job["error"] = None
        self._persist(force=True)

        options = job["options"]
        model_name = str(options.get("model"))
        professional_mode = (
            job.get("preset") == "Motivational Speech"
            and model_name in PROFESSIONAL_MODELS
        )
        professional_longform = professional_mode and options.get("split_text", True)

        directed_segments = []
        if professional_longform:
            retention_positions = [
                int(item.get("word_position", -1))
                for item in (job.get("emotion_summary") or {}).get("placements", [])
                if item.get("source") == "retention-reset" and int(item.get("word_position", -1)) >= 0
            ]
            directed_segments = build_long_form_segments(
                job.get("generation_text") or job["text"],
                max_words=int(options["chunk_words"]),
                heading_detector=is_heading,
                retention_positions=retention_positions,
                age_profile=str(options.get("senior_pace_profile", "70s")),
            )
            chunks = [segment.text for segment in directed_segments]
        else:
            chunks = (
                split_text(
                    job.get("generation_text") or job["text"],
                    int(options["chunk_words"]),
                    prefer_clauses=model_name == "chatterbox-turbo",
                )
                if options.get("split_text", True)
                else [(job.get("generation_text") or job["text"]).strip()]
            )
            chunks = [chunk for chunk in chunks if chunk]
        if not chunks:
            raise ValueError("The script is empty.")

        voice_path = self._resolve_voice(job)
        if professional_mode and voice_path is not None:
            job["stage"] = "Checking reference voice"
            reference_quality = await asyncio.to_thread(analyze_reference_voice, voice_path)
            job["reference_quality"] = reference_quality
            if not reference_quality.get("usable", True) and reference_quality.get("rating") == "Unreadable":
                raise RuntimeError("The selected reference voice could not be decoded. Upload a clean voice sample.")
            if model_name == "chatterbox-turbo" and float(reference_quality.get("duration_seconds", 99)) < 5.1:
                raise RuntimeError("Chatterbox Turbo needs a reference longer than 5 seconds. Use a clean 10–20 second clip.")
            self._persist(force=True)

        title = safe_filename(job["title"] or f"Audio_{job['audio_number']}")
        output_filename = f"Audio_{job['audio_number']}_{title}_{job['id'][:8]}.wav"
        output_path = self.storage.output_path(output_filename)
        output_path.unlink(missing_ok=True)

        completed_words = 0
        started_monotonic = time.monotonic()
        output_file: sf.SoundFile | None = None
        turbo_metric_history: list[dict[str, float]] = []
        speaker_history: list[float] = []
        quality_reports = []
        quality_retries = 0
        caption_asr_words: list[dict[str, Any]] = []
        caption_asr_complete = True
        written_seconds = 0.0
        loop = asyncio.get_running_loop()

        async def render_once(chunk: str, segment, chunk_options: dict[str, Any]) -> tuple[Any, np.ndarray, str, dict[str, float]]:
            spoken_chunk = chunk
            effective_chunk_options = dict(chunk_options)
            if model_name == "chatterbox":
                spoken_chunk, effective_chunk_options = self._original_chunk_direction(chunk, effective_chunk_options)

            result = await loop.run_in_executor(
                self.executor,
                lambda: self.engine.generate(
                    spoken_chunk,
                    model_name=model_name,
                    reference_audio=voice_path,
                    language=effective_chunk_options["language"],
                    options=effective_chunk_options,
                ),
            )
            audio = result.waveform.squeeze().numpy().astype(np.float32, copy=False)
            if professional_mode:
                audio = shape_professional_pauses(audio, result.sample_rate)
                if segment is not None and audio.size:
                    chunk_words_local = count_words(strip_turbo_tags(spoken_chunk))
                    duration_seconds = audio.size / max(result.sample_rate, 1)
                    current_wpm = (chunk_words_local * 60.0 / duration_seconds) if duration_seconds > 0 else 0.0
                    speed_bias = float(options.get("speed_factor", 1.0))
                    target_wpm = float(segment.target_wpm) * speed_bias
                    min_wpm = float(segment.min_wpm) * speed_bias
                    max_wpm = float(segment.max_wpm) * speed_bias
                    tempo = adaptive_tempo_factor(
                        current_wpm=current_wpm,
                        target_wpm=target_wpm,
                        min_wpm=min_wpm,
                        max_wpm=max_wpm,
                    )
                    if abs(tempo - 1.0) > 0.002:
                        audio = await asyncio.to_thread(apply_tempo_array, audio, result.sample_rate, tempo)
                        audio = shape_professional_pauses(audio, result.sample_rate)
                audio = self._level_turbo_chunk(audio)
            raw_metrics = self._audio_metrics(audio, result.sample_rate, max(count_words(strip_turbo_tags(spoken_chunk)), 1))
            return result, audio, spoken_chunk, raw_metrics

        try:
            for index, chunk in enumerate(chunks, start=1):
                segment = directed_segments[index - 1] if professional_longform else None
                if job["cancel_requested"]:
                    job["status"] = "cancelled"
                    job["stage"] = "Cancelled"
                    break

                model_chunk_for_count = strip_turbo_tags(chunk)
                chunk_words = count_words(model_chunk_for_count)
                prior_seconds_per_word = (
                    (time.monotonic() - started_monotonic) / completed_words
                    if completed_words > 0 else 0.36
                )
                expected_chunk = max(4.0, prior_seconds_per_word * max(chunk_words, 1))
                job["stage"] = f"Generating speech {index}/{len(chunks)}"
                ticker = asyncio.create_task(self._smooth_words(job, completed_words, chunk_words, expected_chunk))
                try:
                    result, audio, spoken_chunk, raw_metrics = await render_once(chunk, segment, options)
                finally:
                    ticker.cancel()
                    try:
                        await ticker
                    except asyncio.CancelledError:
                        pass

                # The old Turbo acoustic history remains as a cheap first-line drift
                # detector. Production QC below is the authoritative gate for both models.
                legacy_unstable = False
                if model_name == "chatterbox-turbo" and professional_mode:
                    legacy_unstable = self._turbo_chunk_is_unstable(
                        raw_metrics, turbo_metric_history,
                        intentional_emotion=contains_turbo_tag(chunk),
                    )

                report = None
                if professional_mode and bool(options.get("quality_gate", True)):
                    job["stage"] = f"Quality checking {index}/{len(chunks)}"
                    report = await loop.run_in_executor(
                        self.executor,
                        lambda: self.quality.evaluate(
                            audio,
                            result.sample_rate,
                            spoken_chunk,
                            voice_path if bool(options.get("speaker_consistency", True)) else None,
                            speaker_history,
                        ),
                    )

                needs_retry = legacy_unstable or (report is not None and not report.passed)
                best_result, best_audio, best_spoken, best_metrics, best_report = result, audio, spoken_chunk, raw_metrics, report
                if professional_mode and needs_retry:
                    # At most two focused retries, only for the failed chunk. This raises
                    # production reliability without making every long-form job 2–3x slower.
                    for attempt in range(1, 3):
                        quality_retries += 1
                        retry_options = dict(options)
                        retry_options.update(
                            MOTIVATIONAL_TURBO_RETRY_PROFILE
                            if model_name == "chatterbox-turbo"
                            else MOTIVATIONAL_ORIGINAL_RETRY_PROFILE
                        )
                        retry_options["seed"] = int(options.get("seed", 2025)) + index * 104729 + attempt * 1009
                        retry_options["_quality_retry"] = True
                        # Original gets the same semantic direction, with conservative
                        # retry controls applied before the tag-specific adjustment.
                        job["stage"] = f"Retrying quality issue {index}/{len(chunks)}"
                        retry_result, retry_audio, retry_spoken, retry_metrics = await render_once(chunk, segment, retry_options)
                        retry_report = None
                        if bool(options.get("quality_gate", True)):
                            retry_report = await loop.run_in_executor(
                                self.executor,
                                lambda: self.quality.evaluate(
                                    retry_audio,
                                    retry_result.sample_rate,
                                    retry_spoken,
                                    voice_path if bool(options.get("speaker_consistency", True)) else None,
                                    speaker_history,
                                ),
                            )
                        retry_score = retry_report.score if retry_report is not None else (
                            100.0 - 12.0 * self._chunk_distance_from_history(retry_metrics, turbo_metric_history)
                        )
                        best_score = best_report.score if best_report is not None else (
                            100.0 - 12.0 * self._chunk_distance_from_history(best_metrics, turbo_metric_history)
                        )
                        if retry_score >= best_score:
                            best_result, best_audio, best_spoken, best_metrics, best_report = (
                                retry_result, retry_audio, retry_spoken, retry_metrics, retry_report
                            )
                        if best_report is not None and best_report.passed:
                            break

                result, audio, spoken_chunk, raw_metrics, report = (
                    best_result, best_audio, best_spoken, best_metrics, best_report
                )
                # This is a real production gate, not just a warning badge. If the
                # accepted candidate still fails after focused retries, stop rather
                # than silently shipping a chunk we already know is unreliable.
                if professional_mode and report is not None and not report.passed:
                    quality_reports.append(report)
                    job["quality_summary"] = summarize_quality(quality_reports, quality_retries)
                    if output_file is not None:
                        output_file.close()
                        output_file = None
                    output_path.unlink(missing_ok=True)
                    reason_text = "; ".join(report.reasons[:3]) or "quality verification failed"
                    raise RuntimeError(
                        f"Production Quality Gate rejected chunk {index}/{len(chunks)} after retries: {reason_text}. "
                        "Retry the job or use a cleaner reference voice."
                    )
                if model_name == "chatterbox-turbo":
                    turbo_metric_history.append(raw_metrics)
                if report is not None:
                    quality_reports.append(report)
                    if report.speaker_similarity is not None:
                        speaker_history.append(report.speaker_similarity)

                if output_file is None:
                    output_file = sf.SoundFile(
                        output_path, mode="w", samplerate=result.sample_rate,
                        channels=1, subtype="PCM_16",
                    )

                # Keep raw accepted ASR timing for captions. The final caption text
                # is aligned against the creator-facing script, not the hidden
                # pronunciation copy, so terms such as A1C and 120/80 remain readable.
                audio_duration = audio.size / max(result.sample_rate, 1)
                if report is not None and report.words:
                    for word in report.words:
                        caption_asr_words.append({
                            "start": written_seconds + float(word["start"]),
                            "end": written_seconds + float(word["end"]),
                            "word": str(word["word"]),
                        })
                else:
                    caption_asr_complete = False

                output_file.write(audio)
                written_seconds += audio_duration
                if index < len(chunks):
                    if professional_longform:
                        next_segment = directed_segments[index]
                        pause_ms = next_segment.pause_before_ms
                    else:
                        pause_ms = int(options.get("inter_chunk_pause_ms", 80))
                    pause_seconds = max(0.0, float(pause_ms) / 1000.0)
                    output_file.write(np.zeros(int(result.sample_rate * pause_seconds), dtype=np.float32))
                    written_seconds += pause_seconds

                completed_words += chunk_words
                job["completed_words"] = min(completed_words, job["total_words"])
                job["display_words"] = job["completed_words"]
                elapsed = max(time.monotonic() - started_monotonic, 0.001)
                seconds_per_word = elapsed / max(completed_words, 1)
                remaining_words = max(job["total_words"] - completed_words, 0)
                job["eta_seconds"] = round(seconds_per_word * remaining_words)
                self._update_percent(job)
                self._persist(force=True)

            job["stage"] = "Finalising audio"
            job["percent"] = 98.0
            job["remaining_percent"] = 2.0
            self._persist(force=True)
        finally:
            if output_file is not None:
                output_file.close()

        if job["status"] == "cancelled":
            output_path.unlink(missing_ok=True)
            job["completed_at"] = time.time()
            job["eta_seconds"] = 0
            self._persist(force=True)
            self._signal(job["id"])
            return

        speed_factor = float(options.get("speed_factor", 1.0))
        if not professional_longform and not professional_mode and abs(speed_factor - 1.0) > 0.001:
            job["stage"] = "Applying natural pacing"
            await asyncio.to_thread(self._apply_final_tempo, output_path, speed_factor)

        if professional_mode:
            job["stage"] = "Mastering professional speech"
            job["percent"] = 99.0
            job["remaining_percent"] = 1.0
            self._persist(force=True)
            job["mastering_profile"] = await asyncio.to_thread(master_professional_voice, output_path)
            job["quality_summary"] = summarize_quality(quality_reports, quality_retries)

            if bool(options.get("platform_assets", True)):
                stem = Path(output_filename).stem
                video_filename = f"{stem}.video48.wav"
                srt_filename = f"{stem}.captions.srt"
                vtt_filename = f"{stem}.captions.vtt"
                video_path = self.storage.output_path(video_filename)
                srt_path = self.storage.output_path(srt_filename)
                vtt_path = self.storage.output_path(vtt_filename)
                job["stage"] = "Creating video master and captions"
                await asyncio.to_thread(create_video_master_48k_stereo, output_path, video_path)
                caption_source_text = strip_turbo_tags(job.get("text") or "").strip()
                if caption_asr_complete and caption_asr_words:
                    caption_words = aligned_expected_words(caption_source_text, caption_asr_words, written_seconds)
                else:
                    caption_words = fallback_words(caption_source_text, 0.0, written_seconds)
                await asyncio.to_thread(write_caption_files, caption_words, srt_path, vtt_path)
                job["video_master_filename"] = video_filename
                job["srt_filename"] = srt_filename
                job["vtt_filename"] = vtt_filename
        elif model_name == "chatterbox-turbo":
            job["stage"] = "Normalising Turbo loudness"
            await asyncio.to_thread(self._normalize_turbo_loudness, output_path)

        info = sf.info(output_path)
        job["status"] = "completed"
        job["stage"] = "Completed"
        job["output_filename"] = output_filename
        job["actual_audio_seconds"] = round(float(info.duration), 3)
        job["completed_words"] = job["total_words"]
        job["display_words"] = job["total_words"]
        job["percent"] = 100.0
        job["remaining_percent"] = 0.0
        job["eta_seconds"] = 0
        job["completed_at"] = time.time()
        self._persist(force=True)
        self._signal(job["id"])

