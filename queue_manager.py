from __future__ import annotations

import asyncio
import json
import math
import re
import secrets
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from engine import EngineService
from models import AudioJobCreate
from storage import Storage
from utils import count_words, estimated_audio_seconds, safe_filename

TERMINAL_STATES = {"completed", "failed", "cancelled", "interrupted"}
ACTIVE_STATES = {"queued", "running"}

# Resemble AI's official Chatterbox Turbo Gradio demo labels its input as
# "max chars 300".  Long creator scripts are therefore split only to make the
# official short-input model usable for long-form work.  No text rewriting,
# pronunciation layer, emotion layer, pacing layer, ASR, speaker-QC or rescue
# regeneration is performed.
TURBO_MAX_CHARS = 300
TURBO_INTER_CHUNK_PAUSE_MS = 70
DEFAULT_SENTENCE_END_PAUSE_MS = 280

# Official Turbo Gradio defaults (seed 0 means random there).
OFFICIAL_TURBO_PROFILE: dict[str, Any] = {
    "model": "chatterbox-turbo",
    "language": "en",
    "temperature": 0.8,
    "exaggeration": 0.0,
    "cfg_weight": 0.0,
    "repetition_penalty": 1.2,
    "min_p": 0.0,
    "top_p": 0.95,
    "top_k": 1000,
    "speed_factor": 1.0,
    "split_text": True,
    "chunk_words": 50,        # UI compatibility only; char cap owns splitting
    "inter_chunk_pause_ms": TURBO_INTER_CHUNK_PAUSE_MS,
    "sentence_end_pause_ms": DEFAULT_SENTENCE_END_PAUSE_MS,
    "output_format": "wav",
    "senior_pace_profile": "70s",  # compatibility only; never applied
    "quality_gate": False,
    "speaker_consistency": False,
    "platform_assets": False,
}

# Restored from the user's v1.2.1 Turbo preset. These are intentionally light-touch
# controls only: supported Turbo sampling values, a single pitch-preserving final
# tempo pass, and a small inter-chunk breath. No QC, ASR, emotion, pronunciation,
# prosody, retry/rescue or professional mastering stack is reintroduced.
MOTIVATIONAL_TURBO_PROFILE: dict[str, Any] = {
    **OFFICIAL_TURBO_PROFILE,
    "temperature": 0.72,
    "exaggeration": 0.5,
    "cfg_weight": 0.0,
    "repetition_penalty": 1.2,
    "top_p": 0.95,
    "top_k": 1000,
    "speed_factor": 0.93,
    "inter_chunk_pause_ms": 70,
    "sentence_end_pause_ms": DEFAULT_SENTENCE_END_PAUSE_MS,
}

_TURBO_USER_CONTROLS = {
    # Only the four creator-facing controls requested by the user are accepted
    # from the UI. Turbo's other sampling values stay at stable model defaults.
    "temperature", "exaggeration", "cfg_weight", "speed_factor", "sentence_end_pause_ms",
}

# Keep the user's current louder delivery while making the processing transparent:
# two-pass EBU loudness normalisation only, with no EQ/compression/tempo/prosody work.
FINAL_TARGET_LUFS = -12.5
FINAL_TRUE_PEAK_DBFS = -0.8
FINAL_LRA = 11.0

_SENTENCE_RE = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+")


def _normalise_space(text: str) -> str:
    return " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())


def _split_oversize_piece(piece: str, max_chars: int) -> list[str]:
    """Split one >max_chars span without dropping or rewriting any characters.

    Prefer punctuation and whitespace near the limit.  The rejoined chunks, after
    whitespace normalisation, are guaranteed to equal the source span.
    """
    remaining = piece.strip()
    output: list[str] = []
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        low = max(1, int(max_chars * 0.55))
        candidates = [
            window.rfind(";", low, max_chars + 1),
            window.rfind(":", low, max_chars + 1),
            window.rfind(",", low, max_chars + 1),
            window.rfind(" ", low, max_chars + 1),
        ]
        cut = max(candidates)
        if cut < low:
            cut = window.rfind(" ", 1, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        # Keep punctuation in the left chunk; split whitespace itself away only as
        # normal sentence spacing. Chatterbox already normalises repeated whitespace.
        if remaining[cut:cut + 1] in {";", ":", ","}:
            cut += 1
        left = remaining[:cut].strip()
        if not left:
            cut = max_chars
            left = remaining[:cut].strip()
        output.append(left)
        remaining = remaining[cut:].strip()
    if remaining:
        output.append(remaining)
    return output


def split_turbo_long_text(text: str, max_chars: int = TURBO_MAX_CHARS) -> list[str]:
    """Sentence-safe Chatterbox Turbo chunks capped at ``max_chars``.

    This is deliberately not a narration pipeline. It only groups the creator's
    original text into model-safe calls.  No words, punctuation or wording are
    intentionally added, removed or replaced.
    """
    clean = text.strip()
    if not clean:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", clean) if part.strip()]
    atomic: list[str] = []
    for paragraph in paragraphs:
        sentences = [part.strip() for part in _SENTENCE_RE.split(paragraph) if part.strip()]
        if not sentences:
            sentences = [paragraph]
        for sentence in sentences:
            if len(sentence) <= max_chars:
                atomic.append(sentence)
            else:
                atomic.extend(_split_oversize_piece(sentence, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in atomic:
        candidate = piece if not current else f"{current} {piece}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = piece
    if current:
        chunks.append(current)

    # Defensive integrity gate. If our own splitter ever loses creator text, fail
    # before spending GPU time rather than synthesize an incomplete script.
    if _normalise_space(" ".join(chunks)) != _normalise_space(clean):
        raise RuntimeError("Internal Turbo text splitting changed the script. Generation was stopped before TTS.")
    if any(len(chunk) > max_chars for chunk in chunks):
        raise RuntimeError("Internal Turbo text splitting produced an oversized chunk.")
    return chunks


# Sentence End Pause is a creator control, not a vague long-silence limiter.
# For each Turbo chunk we know how many sentence boundaries are inside the text.
# We therefore normalize only the same number of strongest internal near-silence
# gaps, preserving short breaths and hesitations. Chunk-edge silence is removed so
# the explicit inter-chunk sentence pause is not doubled by model tail silence.
def _sentence_boundary_count(text: str) -> int:
    clean = (text or "").strip()
    if not clean:
        return 0
    parts = [part for part in _SENTENCE_RE.split(clean) if part.strip()]
    return max(0, len(parts) - 1)


def _silence_runs(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_silence_ms: int = 80,
) -> list[tuple[int, int]]:
    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    if data.size == 0 or sample_rate <= 0:
        return []
    frame_ms = 10
    frame_samples = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    frame_count = int(math.ceil(data.size / frame_samples))
    padded = np.pad(data, (0, frame_count * frame_samples - data.size))
    frames = padded.reshape(frame_count, frame_samples)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    speech_reference = float(np.percentile(db, 80))
    silence_threshold = float(np.clip(speech_reference - 30.0, -58.0, -42.0))
    silent = db <= silence_threshold
    min_frames = max(1, int(math.ceil(min_silence_ms / frame_ms)))

    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for idx, is_silent in enumerate(silent):
        if is_silent and run_start is None:
            run_start = idx
        elif not is_silent and run_start is not None:
            if idx - run_start >= min_frames:
                runs.append((run_start * frame_samples, min(data.size, idx * frame_samples)))
            run_start = None
    if run_start is not None and frame_count - run_start >= min_frames:
        runs.append((run_start * frame_samples, data.size))
    return runs


def apply_sentence_end_pause(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    pause_ms: int,
) -> np.ndarray:
    """Normalize actual sentence-ending pauses to the selected duration.

    ``pause_ms`` may be 0..1000. The function changes only the strongest internal
    silence gaps needed for the known number of text sentence boundaries. Shorter
    breath pauses are left intact. Leading/trailing model silence is stripped when
    clearly present so chunk joins can add the requested pause exactly once.
    """
    data = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
    if data.size == 0 or sample_rate <= 0:
        return data
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    target_ms = int(max(0, min(1000, int(pause_ms))))
    runs = _silence_runs(data, sample_rate, min_silence_ms=80)
    if not runs:
        return data

    edge_samples = int(round(sample_rate * 0.060))
    leading_run = next((r for r in runs if r[0] <= edge_samples), None)
    trailing_run = next((r for r in reversed(runs) if r[1] >= data.size - edge_samples), None)

    # Work only with internal gaps for sentence boundaries.
    internal = [
        r for r in runs
        if r is not leading_run and r is not trailing_run
    ]
    boundary_count = _sentence_boundary_count(text)
    # The longest internal gaps are overwhelmingly the punctuation boundaries;
    # shorter breathing gaps remain untouched.
    selected = set(
        sorted(internal, key=lambda r: (r[1] - r[0]), reverse=True)[:boundary_count]
    )

    parts: list[np.ndarray] = []
    cursor = 0
    for run in sorted(runs, key=lambda r: r[0]):
        start_sample, end_sample = run
        if start_sample < cursor or end_sample <= start_sample:
            continue
        parts.append(data[cursor:start_sample])
        if run is leading_run or run is trailing_run:
            # Strip model-generated edge dead air. The queue owns the exact join.
            keep_samples = 0
        elif run in selected:
            keep_samples = int(round(sample_rate * target_ms / 1000.0))
        else:
            # Preserve non-sentence breaths/hesitations exactly as generated.
            parts.append(data[start_sample:end_sample])
            keep_samples = 0
        if keep_samples > 0:
            parts.append(np.zeros(keep_samples, dtype=np.float32))
        cursor = end_sample
    parts.append(data[cursor:])
    return np.concatenate(parts).astype(np.float32, copy=False) if parts else data


class QueueManager:
    """Simple single-GPU Chatterbox Turbo queue.

    v1.6.9 keeps the fast, direct Turbo generation architecture.
    The only output processing after model generation is final loudness normalisation.
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
        # Start from the selected Turbo preset, then honor creator controls.
        # Exaggeration and CFG Weight are preserved as creator inputs. EngineService
        # bridges them to Turbo-supported sampling controls while keeping the fresh
        # generation architecture and all heavy QC/professional layers disabled.
        base = (
            MOTIVATIONAL_TURBO_PROFILE
            if request.preset == "Motivational Speech"
            else OFFICIAL_TURBO_PROFILE
        )
        options = dict(base)
        supplied = request.options.model_dump()
        explicit = set(request.options.model_fields_set)
        for key in _TURBO_USER_CONTROLS:
            if key in explicit:
                options[key] = supplied[key]

        options.update({
            "model": "chatterbox-turbo",
            "language": "en",
            "min_p": 0.0,
            "split_text": True,
            "output_format": "wav",
            "quality_gate": False,
            "speaker_consistency": False,
            "platform_assets": False,
        })
        if int(options.get("seed", 0)) <= 0:
            options["seed"] = secrets.randbelow(2_147_483_646) + 1
        return options

    @staticmethod
    def _apply_final_tempo(output_path: Path, speed_factor: float) -> None:
        """Apply one pitch-preserving tempo pass when the creator changes Speed Factor."""
        if abs(speed_factor - 1.0) <= 0.001:
            return
        temp_path = output_path.with_name(output_path.stem + ".tempo.wav")
        temp_path.unlink(missing_ok=True)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(output_path), "-filter:a", f"atempo={speed_factor:.6f}",
                    "-c:a", "pcm_s16le", str(temp_path),
                ],
                check=True,
            )
            temp_path.replace(output_path)
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError("Speed Factor requires a working FFmpeg atempo filter.") from error

    async def create(self, request: AudioJobCreate, enqueue: bool = True) -> dict[str, Any]:
        total_words = count_words(request.text)
        if total_words == 0:
            raise ValueError("The script is empty.")
        job_id = uuid.uuid4().hex
        now = time.time()
        options = self._effective_options(request)
        job = {
            "id": job_id,
            "preset": request.preset or "Motivational Speech",
            "generation_mode": "standard",
            "auto_emotion": False,
            "monitor_dismissed": False,
            "audio_number": request.audio_number,
            "title": request.title.strip(),
            "text": request.text,
            "voice_mode": request.voice_mode,
            "voice_filename": request.voice_filename,
            "options": options,
            "status": "queued" if enqueue else "draft",
            "stage": "Waiting in queue" if enqueue else "Draft",
            "total_words": total_words,
            "completed_words": 0,
            "display_words": 0,
            "percent": 0.0,
            "remaining_percent": 100.0,
            "eta_seconds": None,
            "estimated_audio_seconds": round(
                estimated_audio_seconds(total_words, float(options.get("speed_factor", 1.0))), 1
            ),
            "actual_audio_seconds": None,
            "output_filename": None,
            "video_master_filename": None,
            "srt_filename": None,
            "vtt_filename": None,
            "quality_summary": None,
            "reference_quality": None,
            "mastering_profile": None,
            "prosody_summary": None,
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
        return [await self.create(request, enqueue=True) for request in requests]

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
            job["stage"] = "Cancellation requested; waiting for the current Turbo pass"
        self._persist(force=True)
        return self.public_job(job)

    async def dismiss_from_monitor(self, job_id: str) -> dict[str, Any]:
        job = self.get_raw(job_id)
        if job.get("status") in ACTIVE_STATES:
            raise RuntimeError("Active jobs cannot be hidden from the Queue Monitor.")
        job["monitor_dismissed"] = True
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

    async def _smooth_words(self, job: dict[str, Any], base_words: int, active_words: int, expected_seconds: float) -> None:
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

    @staticmethod
    def _normalise_loudness(output_path: Path) -> dict[str, float | str]:
        """Keep current loud volume with loudness-only post processing.

        Two-pass loudnorm is retained solely because the user explicitly wants the
        current output level. No EQ, compression, de-essing, tempo, silence shaping,
        prosody processing or per-chunk levelling is used.
        """
        info = sf.info(output_path)
        temp_path = output_path.with_name(output_path.stem + ".volume.wav")
        temp_path.unlink(missing_ok=True)
        base = (
            f"loudnorm=I={FINAL_TARGET_LUFS:.1f}:TP={FINAL_TRUE_PEAK_DBFS:.1f}:"
            f"LRA={FINAL_LRA:.1f}:print_format=json"
        )
        try:
            first = subprocess.run(
                ["ffmpeg", "-hide_banner", "-nostats", "-i", str(output_path), "-af", base, "-f", "null", "-"],
                check=True, capture_output=True, text=True,
            )
            stderr = first.stderr
            start = stderr.rfind("{")
            end = stderr.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("FFmpeg loudness analysis did not return JSON.")
            measured = json.loads(stderr[start:end + 1])
            second_filter = (
                f"loudnorm=I={FINAL_TARGET_LUFS:.1f}:TP={FINAL_TRUE_PEAK_DBFS:.1f}:LRA={FINAL_LRA:.1f}:"
                f"measured_I={float(measured['input_i']):.2f}:"
                f"measured_LRA={float(measured['input_lra']):.2f}:"
                f"measured_TP={float(measured['input_tp']):.2f}:"
                f"measured_thresh={float(measured['input_thresh']):.2f}:"
                f"offset={float(measured['target_offset']):.2f}:linear=true:print_format=summary"
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(output_path),
                    "-af", second_filter, "-ar", str(info.samplerate), "-ac", "1", "-c:a", "pcm_s16le",
                    str(temp_path),
                ],
                check=True,
            )
            temp_path.replace(output_path)
            return {
                "profile": "loudness-only",
                "target_lufs": FINAL_TARGET_LUFS,
                "true_peak_dbfs": FINAL_TRUE_PEAK_DBFS,
            }
        except (FileNotFoundError, subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as error:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError("Final volume boost requires a working FFmpeg loudnorm filter.") from error

    async def _process(self, job: dict[str, Any]) -> None:
        job["status"] = "running"
        job["stage"] = "Preparing Chatterbox Turbo"
        job["started_at"] = time.time()
        job["error"] = None
        self._persist(force=True)

        options = job["options"]
        chunks = split_turbo_long_text(job["text"], TURBO_MAX_CHARS)
        if not chunks:
            raise ValueError("The script is empty.")

        voice_path = self._resolve_voice(job)
        title = safe_filename(job["title"] or f"Audio_{job['audio_number']}")
        output_filename = f"Audio_{job['audio_number']}_{title}_{job['id'][:8]}.wav"
        output_path = self.storage.output_path(output_filename)
        output_path.unlink(missing_ok=True)

        completed_words = 0
        started_monotonic = time.monotonic()
        output_file: sf.SoundFile | None = None
        loop = asyncio.get_running_loop()

        try:
            for index, chunk in enumerate(chunks, start=1):
                if job["cancel_requested"]:
                    job["status"] = "cancelled"
                    job["stage"] = "Cancelled"
                    break

                chunk_words = count_words(chunk)
                prior_seconds_per_word = (
                    (time.monotonic() - started_monotonic) / completed_words
                    if completed_words > 0 else 0.30
                )
                expected_chunk = max(2.0, prior_seconds_per_word * max(chunk_words, 1))
                job["stage"] = f"Generating speech {index}/{len(chunks)}"
                ticker = asyncio.create_task(self._smooth_words(job, completed_words, chunk_words, expected_chunk))

                # Official demo uses random seed when seed=0. Our adapter accepts an
                # explicit seed, so vary it per call rather than forcing identical
                # random state across every long-form chunk.
                chunk_options = dict(options)
                chunk_options["seed"] = (int(options["seed"]) + index * 1009) % 2_147_483_647 or 1
                try:
                    result = await loop.run_in_executor(
                        self.executor,
                        lambda c=chunk, o=chunk_options: self.engine.generate(
                            c,
                            model_name="chatterbox-turbo",
                            reference_audio=voice_path,
                            language="en",
                            options=o,
                        ),
                    )
                finally:
                    ticker.cancel()
                    try:
                        await ticker
                    except asyncio.CancelledError:
                        pass

                audio = result.waveform.squeeze().numpy().astype(np.float32, copy=False)
                if audio.size == 0 or not np.isfinite(audio).all():
                    raise RuntimeError(f"Chatterbox Turbo returned invalid audio for chunk {index}/{len(chunks)}.")

                # Apply the creator-selected pause to real sentence boundaries.
                # Unlike v1.6.7 this is measurable at 0, 350, 1000ms etc.
                sentence_end_pause_ms = int(options.get(
                    "sentence_end_pause_ms", DEFAULT_SENTENCE_END_PAUSE_MS
                ))
                audio = apply_sentence_end_pause(
                    audio, result.sample_rate, chunk, sentence_end_pause_ms
                )

                if output_file is None:
                    output_file = sf.SoundFile(
                        output_path, mode="w", samplerate=result.sample_rate,
                        channels=1, subtype="PCM_16",
                    )
                elif int(output_file.samplerate) != int(result.sample_rate):
                    raise RuntimeError("Chatterbox Turbo changed sample rate during one job.")

                output_file.write(audio)
                if index < len(chunks):
                    # If the chunk ends at sentence punctuation, the creator's
                    # Sentence End Pause owns the join. Oversize mid-sentence splits
                    # use only a tiny continuity gap.
                    clean_chunk = chunk.rstrip().rstrip('"\'”’)]')
                    ends_sentence = bool(clean_chunk) and clean_chunk[-1:] in {".", "!", "?"}
                    pause_ms = sentence_end_pause_ms if ends_sentence else 40
                    if pause_ms > 0:
                        pause = int(result.sample_rate * pause_ms / 1000.0)
                        output_file.write(np.zeros(pause, dtype=np.float32))

                completed_words += chunk_words
                job["completed_words"] = min(completed_words, job["total_words"])
                job["display_words"] = job["completed_words"]
                elapsed = max(time.monotonic() - started_monotonic, 0.001)
                seconds_per_word = elapsed / max(completed_words, 1)
                remaining_words = max(job["total_words"] - completed_words, 0)
                job["eta_seconds"] = round(seconds_per_word * remaining_words)
                self._update_percent(job)
                self._persist(force=True)

            job["stage"] = "Finalising audio volume"
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
        if abs(speed_factor - 1.0) > 0.001:
            job["stage"] = "Applying natural pacing"
            job["percent"] = 99.0
            job["remaining_percent"] = 1.0
            self._persist(force=True)
            await asyncio.to_thread(self._apply_final_tempo, output_path, speed_factor)

        # Keep the user's requested louder final level after the optional tempo pass.
        job["mastering_profile"] = await asyncio.to_thread(self._normalise_loudness, output_path)
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
