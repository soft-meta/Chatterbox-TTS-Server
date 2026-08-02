from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Awaitable, Callable

from avatar_engine import AvatarCancelled, AvatarEngineService
from models import VideoJobCreate
from storage import Storage

TERMINAL_STATES = {"completed", "failed", "cancelled", "interrupted"}
ACTIVE_STATES = {"queued", "running"}

BeforeHook = Callable[[], Awaitable[Any]]
AfterHook = Callable[[Any], Awaitable[None]]


class VideoQueueManager:
    def __init__(
        self,
        engine: AvatarEngineService,
        storage: Storage,
        before_process: BeforeHook | None = None,
        after_process: AfterHook | None = None,
    ) -> None:
        self.engine = engine
        self.storage = storage
        self.before_process = before_process
        self.after_process = after_process
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.jobs: dict[str, dict[str, Any]] = self.storage.load_video_jobs()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="softmeta-avatar")
        self.worker_task: asyncio.Task | None = None
        self._last_persist = 0.0
        for job in self.jobs.values():
            if job.get("status") in ACTIVE_STATES:
                job["status"] = "interrupted"
                job["stage"] = "Runtime restarted before this video completed."
                job["completed_at"] = time.time()
        self._persist(force=True)

    def _persist(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_persist >= 1.0:
            self.storage.save_video_jobs(self.jobs)
            self._last_persist = now

    async def start(self) -> None:
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker(), name="softmeta-video-queue")

    async def stop(self) -> None:
        self.engine.cancel_active()
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=False, cancel_futures=True)
        self._persist(force=True)

    def public_job(self, job: dict[str, Any]) -> dict[str, Any]:
        data = dict(job)
        data.pop("cancel_requested", None)
        data.pop("avatar_path", None)
        data.pop("audio_path", None)
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

    async def create(
        self,
        request: VideoJobCreate,
        avatar_path: Path,
        audio_path: Path,
        audio_label: str,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = time.time()
        job = {
            "id": job_id,
            **request.model_dump(),
            "avatar_path": str(avatar_path),
            "audio_path": str(audio_path),
            "audio_label": audio_label,
            "status": "queued",
            "stage": "Waiting for avatar GPU",
            "percent": 0.0,
            "eta_seconds": None,
            "output_filename": None,
            "output_size": None,
            "duration": None,
            "backend": None,
            "backend_label": None,
            "segments": None,
            "quality_report": None,
            "log_filename": None,
            "error": None,
            "cancel_requested": False,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
        }
        self.jobs[job_id] = job
        self._persist(force=True)
        await self.queue.put(job_id)
        return self.public_job(job)

    async def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get_raw(job_id)
        if job["status"] in TERMINAL_STATES:
            return self.public_job(job)
        job["cancel_requested"] = True
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["stage"] = "Cancelled before rendering"
            job["completed_at"] = time.time()
        else:
            job["stage"] = "Cancellation requested; stopping the current avatar pass"
        self._persist(force=True)
        return self.public_job(job)

    async def delete(self, job_id: str, delete_file: bool = True) -> None:
        job = self.get_raw(job_id)
        if job.get("status") in ACTIVE_STATES:
            raise RuntimeError("Active video jobs cannot be removed.")
        if delete_file and job.get("output_filename"):
            self.storage.delete_video_artifacts(job["output_filename"], job_id)
        self.jobs.pop(job_id, None)
        self._persist(force=True)

    async def clear(self, delete_files: bool = True) -> None:
        if self.has_active_jobs():
            raise RuntimeError("Wait for the active video job to finish or cancel it first.")
        if delete_files:
            for job in self.jobs.values():
                if job.get("output_filename"):
                    self.storage.delete_video_artifacts(job["output_filename"], job["id"])
        self.jobs.clear()
        self._persist(force=True)

    async def _worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                job = self.jobs.get(job_id)
                if not job or job.get("status") == "cancelled":
                    continue
                await self._process(job)
            except asyncio.CancelledError:
                raise
            finally:
                self.queue.task_done()

    async def _process(self, job: dict[str, Any]) -> None:
        hook_state: Any = None
        job["status"] = "running"
        job["stage"] = "Preparing avatar environment"
        job["started_at"] = time.time()
        job["error"] = None
        self._persist(force=True)
        try:
            if self.before_process:
                hook_state = await self.before_process()
            loop = asyncio.get_running_loop()

            def progress(percent: float, stage: str, eta: float | None) -> None:
                job["percent"] = round(max(0.0, min(100.0, percent)), 1)
                job["stage"] = stage
                job["eta_seconds"] = None if eta is None else max(0, round(eta))
                self._persist()

            def cancelled() -> bool:
                return bool(job.get("cancel_requested"))

            result = await loop.run_in_executor(
                self.executor,
                lambda: self.engine.render(
                    job,
                    Path(job["avatar_path"]),
                    Path(job["audio_path"]),
                    progress,
                    cancelled,
                ),
            )
            job.update(result)
            job["status"] = "completed"
            job["stage"] = "Video completed"
            job["percent"] = 100.0
            job["eta_seconds"] = 0
            job["completed_at"] = time.time()
        except AvatarCancelled as error:
            job["status"] = "cancelled"
            job["stage"] = "Video generation cancelled"
            job["error"] = str(error)
            job["completed_at"] = time.time()
            job["eta_seconds"] = 0
        except Exception as error:  # noqa: BLE001
            job["status"] = "failed"
            job["stage"] = "Video generation failed"
            job["error"] = f"{type(error).__name__}: {error}"
            job["completed_at"] = time.time()
            job["eta_seconds"] = 0
        finally:
            if self.after_process:
                try:
                    await self.after_process(hook_state)
                except Exception as error:  # noqa: BLE001
                    job["restore_warning"] = f"Could not restore TTS model: {type(error).__name__}: {error}"
            self._persist(force=True)
