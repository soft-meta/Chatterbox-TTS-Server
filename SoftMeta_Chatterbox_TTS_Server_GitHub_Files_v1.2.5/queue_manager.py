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
from emotion_director import analyze_serious_senior_advisor, contains_turbo_tag, strip_turbo_tags
from models import AudioJobCreate
from storage import Storage
from utils import count_words, estimated_audio_seconds, safe_filename, split_text

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
    # Keep the senior-advisor pace, but use shorter chunks than v1.2.2. 85 words
    # is a stability/speed compromise: it limits within-chunk prosody drift while
    # retaining most of Turbo's long-form speed advantage on an L4.
    "speed_factor": 0.93,
    "chunk_words": 85,
    "inter_chunk_pause_ms": 140,
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
            and effective_options.get("model") == "chatterbox-turbo"
        ):
            emotion_analysis = analyze_serious_senior_advisor(request.text)
            generation_text = emotion_analysis.tagged_text
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
            split_text(
                job.get("generation_text") or job["text"],
                int(options["chunk_words"]),
                prefer_clauses=options.get("model") == "chatterbox-turbo",
            )
            if options.get("split_text", True)
            else [(job.get("generation_text") or job["text"]).strip()]
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
        turbo_metric_history: list[dict[str, float]] = []

        try:
            for index, chunk in enumerate(chunks, start=1):
                if job["cancel_requested"]:
                    job["status"] = "cancelled"
                    job["stage"] = "Cancelled"
                    break

                chunk_words = count_words(strip_turbo_tags(chunk))
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

                if options.get("model") == "chatterbox-turbo":
                    raw_metrics = self._audio_metrics(audio, result.sample_rate, chunk_words)
                    if (
                        job.get("preset") == "Motivational Speech"
                        and self._turbo_chunk_is_unstable(
                            raw_metrics,
                            turbo_metric_history,
                            intentional_emotion=contains_turbo_tag(chunk),
                        )
                    ):
                        # Retry only anomalous chunks, not every chunk, so long-form
                        # speed remains close to Turbo's normal throughput.
                        retry_options = dict(options)
                        retry_options.update(MOTIVATIONAL_TURBO_RETRY_PROFILE)
                        retry_options["seed"] = int(options.get("seed", 2025)) + index * 104729
                        logger.warning(
                            "Turbo stability guard retrying chunk %s: rms=%.1f dBFS, %.3f sec/word",
                            index, raw_metrics["rms_dbfs"], raw_metrics["seconds_per_word"],
                        )
                        retry_result = await loop.run_in_executor(
                            self.executor,
                            lambda: self.engine.generate(
                                chunk,
                                model_name=options["model"],
                                reference_audio=voice_path,
                                language=options["language"],
                                options=retry_options,
                            ),
                        )
                        retry_audio = retry_result.waveform.squeeze().numpy().astype(np.float32, copy=False)
                        retry_metrics = self._audio_metrics(retry_audio, retry_result.sample_rate, chunk_words)
                        if (
                            not turbo_metric_history
                            or self._chunk_distance_from_history(retry_metrics, turbo_metric_history)
                            <= self._chunk_distance_from_history(raw_metrics, turbo_metric_history)
                        ):
                            result = retry_result
                            audio = retry_audio
                            raw_metrics = retry_metrics
                    turbo_metric_history.append(raw_metrics)
                    audio = self._level_turbo_chunk(audio)

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
