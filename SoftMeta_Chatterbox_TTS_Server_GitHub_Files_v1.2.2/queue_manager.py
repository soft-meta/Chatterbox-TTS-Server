from __future__ import annotations

import asyncio
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
from utils import count_words, estimated_audio_seconds, safe_filename, split_text

TERMINAL_STATES = {"completed", "failed", "cancelled", "interrupted"}
ACTIVE_STATES = {"queued", "running"}

MOTIVATIONAL_TURBO_PROFILE: dict[str, Any] = {
    # Chatterbox Turbo ignores CFG, exaggeration and min_p. Keep them neutral.
    # Supported sampling values are intentionally conservative for a calm,
    # understandable senior-advisor / tutorial delivery.
    "temperature": 0.72,
    "exaggeration": 0.0,
    "cfg_weight": 0.0,
    "repetition_penalty": 1.2,
    "min_p": 0.0,
    "top_p": 0.90,
    "top_k": 1000,
    # 0.93 ~= 7.5% slower than the generated waveform. Final tempo adjustment
    # is pitch-preserving through FFmpeg atempo, so the voice does not become
    # artificially deeper.
    "speed_factor": 0.93,
    "chunk_words": 105,
    "inter_chunk_pause_ms": 140,
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
        options = request.options.model_dump()
        if (
            request.preset == "Motivational Speech"
            and options.get("model") == "chatterbox-turbo"
        ):
            options.update(MOTIVATIONAL_TURBO_PROFILE)
        return options

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
    def _normalize_turbo_loudness(output_path: Path) -> None:
        """Master Turbo speech to a clear web-friendly loudness without clipping.

        Turbo can return noticeably lower-level waveforms than Original.  The final
        WAV is therefore normalised to a speech-friendly -16 LUFS target with a
        -1 dB true-peak ceiling.  The original sample rate and mono layout are kept.
        """
        info = sf.info(output_path)
        temp_path = output_path.with_name(output_path.stem + ".loud.wav")
        temp_path.unlink(missing_ok=True)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(output_path),
                    "-filter:a", "loudnorm=I=-16.0:TP=-1.0:LRA=7.0",
                    "-ar", str(info.samplerate),
                    "-ac", "1",
                    "-c:a", "pcm_s16le",
                    str(temp_path),
                ],
                check=True,
            )
            temp_path.replace(output_path)
        except (FileNotFoundError, subprocess.CalledProcessError):
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Turbo loudness normalisation requires FFmpeg. "
                "Install ffmpeg and try again."
            )

    async def create(self, request: AudioJobCreate, enqueue: bool = True) -> dict[str, Any]:
        total_words = count_words(request.text)
        if total_words == 0:
            raise ValueError("The script is empty.")
        job_id = uuid.uuid4().hex
        now = time.time()
        effective_options = self._effective_options(request)
        job = {
            "id": job_id,
            "preset": request.preset,
            "audio_number": request.audio_number,
            "title": request.title.strip(),
            "text": request.text,
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
        chunks = (
            split_text(job["text"], int(options["chunk_words"]))
            if options.get("split_text", True)
            else [job["text"].strip()]
        )
        chunks = [chunk for chunk in chunks if chunk]
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

        try:
            for index, chunk in enumerate(chunks, start=1):
                if job["cancel_requested"]:
                    job["status"] = "cancelled"
                    job["stage"] = "Cancelled"
                    break

                chunk_words = count_words(chunk)
                prior_seconds_per_word = (
                    (time.monotonic() - started_monotonic) / completed_words
                    if completed_words > 0
                    else 0.36
                )
                expected_chunk = max(4.0, prior_seconds_per_word * max(chunk_words, 1))
                job["stage"] = "Generating speech"
                ticker = asyncio.create_task(
                    self._smooth_words(job, completed_words, chunk_words, expected_chunk)
                )

                loop = asyncio.get_running_loop()
                try:
                    result = await loop.run_in_executor(
                        self.executor,
                        lambda: self.engine.generate(
                            chunk,
                            model_name=options["model"],
                            reference_audio=voice_path,
                            language=options["language"],
                            options=options,
                        ),
                    )
                finally:
                    ticker.cancel()
                    try:
                        await ticker
                    except asyncio.CancelledError:
                        pass

                audio = result.waveform.squeeze().numpy().astype(np.float32, copy=False)
                if output_file is None:
                    output_file = sf.SoundFile(
                        output_path,
                        mode="w",
                        samplerate=result.sample_rate,
                        channels=1,
                        subtype="PCM_16",
                    )
                output_file.write(audio)
                if index < len(chunks):
                    pause_seconds = max(0.0, float(options.get("inter_chunk_pause_ms", 80)) / 1000.0)
                    output_file.write(
                        np.zeros(int(result.sample_rate * pause_seconds), dtype=np.float32)
                    )

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
        if abs(speed_factor - 1.0) > 0.001:
            job["stage"] = "Applying natural pacing"
            job["percent"] = 98.7
            job["remaining_percent"] = 1.3
            self._persist(force=True)
            await asyncio.to_thread(self._apply_final_tempo, output_path, speed_factor)

        # Chatterbox Turbo often produces a quieter waveform than Original.
        # Normalise only Turbo outputs so Original keeps its established sound.
        if options.get("model") == "chatterbox-turbo":
            job["stage"] = "Balancing Turbo loudness"
            job["percent"] = 99.4
            job["remaining_percent"] = 0.6
            self._persist(force=True)
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
