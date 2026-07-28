from __future__ import annotations

import asyncio
import math
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

TERMINAL_STATES = {"completed", "failed", "cancelled"}


class QueueManager:
    def __init__(self, engine: EngineService, storage: Storage) -> None:
        self.engine = engine
        self.storage = storage
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.jobs: dict[str, dict[str, Any]] = self.storage.load_jobs()
        self.worker_task: asyncio.Task | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="softmeta-tts")
        self._waiters: dict[str, asyncio.Event] = {}

        for job in self.jobs.values():
            if job.get("status") in {"running", "queued"}:
                job["status"] = "interrupted"
                job["stage"] = "Runtime restarted before completion."
        self._persist()

    def _persist(self) -> None:
        self.storage.save_jobs(self.jobs)

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

    def list_jobs(self) -> list[dict[str, Any]]:
        return sorted(self.jobs.values(), key=lambda item: item.get("created_at", 0.0))

    def get(self, job_id: str) -> dict[str, Any]:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    async def create(self, request: AudioJobCreate, enqueue: bool = True) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        total_words = count_words(request.text)
        now = time.time()
        job = {
            "id": job_id,
            "audio_number": request.audio_number,
            "title": request.title.strip(),
            "text": request.text,
            "voice_mode": request.voice_mode,
            "voice_filename": request.voice_filename,
            "options": request.options.model_dump(),
            "status": "queued" if enqueue else "draft",
            "stage": "Waiting in queue" if enqueue else "Draft",
            "total_words": total_words,
            "completed_words": 0,
            "display_words": 0,
            "percent": 0.0,
            "remaining_percent": 100.0,
            "eta_seconds": None,
            "estimated_audio_seconds": estimated_audio_seconds(
                total_words,
                request.options.speed_factor,
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
        self._persist()
        if enqueue:
            await self.queue.put(job_id)
        return job

    async def create_many(self, requests: list[AudioJobCreate]) -> list[dict[str, Any]]:
        created = []
        for request in requests:
            created.append(await self.create(request, enqueue=True))
        return created

    async def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job["status"] in TERMINAL_STATES:
            return job
        job["cancel_requested"] = True
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["stage"] = "Cancelled before generation"
            job["completed_at"] = time.time()
            self._signal(job_id)
        self._persist()
        return job

    async def wait(self, job_id: str, timeout: float | None = None) -> dict[str, Any]:
        job = self.get(job_id)
        if job["status"] in TERMINAL_STATES:
            return job
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
                    self._persist()
                    self._signal(job_id)
            finally:
                self.queue.task_done()

    def _resolve_voice(self, job: dict[str, Any]) -> Path | None:
        if job["voice_mode"] == "default":
            return None
        return self.storage.voice_path(job["voice_mode"], job["voice_filename"])

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
                    job["total_words"] - 1,
                    base_words + int(active_words * fraction),
                )
                self._update_percent(job)
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _update_percent(job: dict[str, Any], cap: float = 95.0) -> None:
        total = max(int(job["total_words"]), 1)
        percent = min(cap, (float(job["display_words"]) / total) * 95.0)
        job["percent"] = round(percent, 1)
        job["remaining_percent"] = round(max(0.0, 100.0 - percent), 1)

    async def _process(self, job: dict[str, Any]) -> None:
        job["status"] = "running"
        job["stage"] = "Preparing text and voice"
        job["started_at"] = time.time()
        job["error"] = None
        self._persist()

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
        if output_path.exists():
            output_path.unlink()

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
                prior_rate = (
                    (time.monotonic() - started_monotonic) / completed_words
                    if completed_words > 0
                    else 0.35
                )
                expected_chunk = max(4.0, prior_rate * max(chunk_words, 1))
                job["stage"] = f"Generating audio {index} of {len(chunks)}"
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
                    output_file.write(np.zeros(int(result.sample_rate * 0.08), dtype=np.float32))

                completed_words += chunk_words
                job["completed_words"] = min(completed_words, job["total_words"])
                job["display_words"] = job["completed_words"]
                elapsed = max(time.monotonic() - started_monotonic, 0.001)
                seconds_per_word = elapsed / max(completed_words, 1)
                remaining_words = max(job["total_words"] - completed_words, 0)
                job["eta_seconds"] = round(seconds_per_word * remaining_words)
                self._update_percent(job)
                self._persist()

            job["stage"] = "Finalising WAV file"
            job["percent"] = 98.0
            job["remaining_percent"] = 2.0
        finally:
            if output_file is not None:
                output_file.close()

        if job["status"] == "cancelled":
            output_path.unlink(missing_ok=True)
            job["completed_at"] = time.time()
            job["eta_seconds"] = 0
            self._persist()
            self._signal(job["id"])
            return

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
        self._persist()
        self._signal(job["id"])
