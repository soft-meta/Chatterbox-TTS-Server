from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import ROOT, load_config
from engine import EngineService
from emotion_director import analyze_serious_senior_advisor
from reference_quality import analyze_reference_voice
from models import (
    AudioJobCreate,
    CutRequest,
    GenerateAllRequest,
    EmotionAnalyzeRequest,
    ModelLoadRequest,
    OpenAITTSRequest,
    RemoveJobsRequest,
    PerformanceFeedbackRequest,
)
from queue_manager import QueueManager
from storage import AUDIO_EXTENSIONS, Storage
from utils import safe_filename

APP_NAME = "SoftMeta Chatterbox TTS Server"
APP_VERSION = "1.5.3"
logger = logging.getLogger("softmeta.chatterbox")

config = load_config()
storage = Storage()
engine = EngineService(device=config["tts_engine"]["device"])
queue = QueueManager(engine=engine, storage=storage)
model_load_task: asyncio.Task | None = None

MODELS = [
    {
        "id": "chatterbox",
        "name": "Chatterbox Original (English)",
        "badge": "Original",
        "description": "Default English narration engine. Uses Original CFG and Exaggeration controls with the full Professional Speech Pipeline.",
    },
    {
        "id": "chatterbox-turbo",
        "name": "Chatterbox Turbo (English)",
        "badge": "Turbo",
        "description": "Optional faster English engine. Uses the same Professional Speech Pipeline when selected.",
    },
    {
        "id": "chatterbox-nano",
        "name": "Chatterbox Nano (English)",
        "badge": "Nano",
        "description": "Compact English model when supported by the installed engine.",
    },
    {
        "id": "chatterbox-multilingual",
        "name": "Chatterbox Multilingual",
        "badge": "Multilingual",
        "description": "Multilingual generation when supported by the installed engine.",
    },
]

PRESETS = [
    {
        "name": "Motivational Speech",
        "description": "Production-grade senior-advisor narration for Original or Turbo with pronunciation control, quality verification, speaker consistency, age-aware pacing, captions and platform mastering.",
        "language": "en",
        "temperature": 0.72,
        "exaggeration": 0.58,
        "cfg_weight": 0.35,
        "repetition_penalty": 1.2,
        "min_p": 0.05,
        "top_p": 1.0,
        "top_k": 1000,
        "speed_factor": 1.0,
        "seed": 2025,
        "split_text": True,
        "chunk_words": 85,
        "output_format": "wav",
        "senior_pace_profile": "70s",
        "quality_gate": True,
        "speaker_consistency": True,
        "platform_assets": True,
        "turbo_overrides": {
            # Turbo does not support CFG/exaggeration/min_p. These values are
            # deliberately neutral so the model does not emit ignored-control warnings.
            # The supported sampling controls plus a pitch-preserving final tempo pass
            # produce a calm senior-advisor / tutorial delivery.
            "temperature": 0.60,
            "exaggeration": 0.0,
            "cfg_weight": 0.0,
            "repetition_penalty": 1.18,
            "min_p": 0.0,
            "top_p": 0.85,
            "top_k": 600,
            "speed_factor": 1.0,
            "chunk_words": 85,
        },
    },
    {
        "name": "Natural Conversation",
        "description": "Balanced, calm and natural delivery for general narration.",
        "language": "en",
        "temperature": 0.8,
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "repetition_penalty": 1.2,
        "min_p": 0.05,
        "top_p": 1.0,
        "top_k": 1000,
        "speed_factor": 1.0,
        "seed": 2025,
        "split_text": True,
        "chunk_words": 90,
        "output_format": "wav",
    },
    {
        "name": "Calm Storytelling",
        "description": "Measured delivery with gentle emotion for long-form stories.",
        "language": "en",
        "temperature": 0.75,
        "exaggeration": 0.45,
        "cfg_weight": 0.45,
        "repetition_penalty": 1.2,
        "min_p": 0.05,
        "top_p": 1.0,
        "top_k": 1000,
        "speed_factor": 0.96,
        "seed": 2025,
        "split_text": True,
        "chunk_words": 85,
        "output_format": "wav",
    },
]


async def _load_model_in_background(model_name: str) -> None:
    global model_load_task
    try:
        await asyncio.get_running_loop().run_in_executor(queue.executor, engine.load, model_name)
        logger.info("Loaded model %s on %s", model_name, engine.device)
    except Exception:
        logger.exception("Model loading failed")
    finally:
        model_load_task = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model_load_task
    await queue.start()
    if config["server"].get("auto_load_model", True):
        default_model = config["tts_engine"]["default_model"]
        model_load_task = asyncio.create_task(_load_model_in_background(default_model))
    yield
    if model_load_task and not model_load_task.done():
        model_load_task.cancel()
    await queue.stop()


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="SoftMeta self-hosted Chatterbox TTS server and studio.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=ROOT / "ui"), name="static")
app.mount("/outputs", StaticFiles(directory=storage.outputs), name="outputs")


@app.exception_handler(Exception)
async def unexpected_error(_: Request, error: Exception) -> JSONResponse:
    logger.exception("Unhandled server error")
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"{type(error).__name__}: {error}",
            "message": "The server could not complete this request. Check the Colab log for details.",
        },
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(
        (ROOT / "ui" / "index.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "app": APP_NAME,
        "version": APP_VERSION,
        "engine": engine.status(),
        "queue_size": queue.queue.qsize(),
        "active_jobs": queue.has_active_jobs(),
    }


@app.get("/api/ui/initial-data")
def initial_data() -> dict[str, Any]:
    return {
        "app": {"name": APP_NAME, "version": APP_VERSION},
        "models": MODELS,
        "active_model": engine.loaded_model or config["tts_engine"]["default_model"],
        "engine": engine.status(),
        "predefined_voices": storage.list_audio(storage.voices),
        "reference_voices": storage.list_audio(storage.references),
        "presets": PRESETS,
        "defaults": config["generation_defaults"],
        "jobs": queue.list_jobs(),
        "limits": {
            "max_audio_tabs": 5,
            "max_voice_upload_mb": 100,
        },
        "production_quality": {
            "models": ["chatterbox", "chatterbox-turbo"],
            "asr_model": "small.en",
            "speaker_verifier": "speechbrain/spkrec-ecapa-voxceleb",
            "video_master_sample_rate": 48000,
        },
    }




@app.post("/api/emotion/analyze")
def analyze_emotion(request: EmotionAnalyzeRequest) -> dict[str, Any]:
    """Preview the conservative emotion direction used by Original and Turbo."""
    analysis = analyze_serious_senior_advisor(request.text)
    return analysis.public_summary(include_text=True)


@app.get("/api/model-info")
def model_info() -> dict[str, Any]:
    return engine.status()


@app.post("/api/model/load")
async def load_model(request: ModelLoadRequest) -> dict[str, Any]:
    global model_load_task
    if queue.has_active_jobs():
        raise HTTPException(409, "Wait for audio generation to finish before changing models.")
    if model_load_task and not model_load_task.done():
        raise HTTPException(409, "A model is already loading.")
    try:
        await asyncio.get_running_loop().run_in_executor(queue.executor, engine.load, request.model)
    except Exception as error:
        raise HTTPException(500, f"Model loading failed: {type(error).__name__}: {error}") from error
    return engine.status()


@app.post("/api/model/unload")
async def unload_model() -> dict[str, Any]:
    if queue.has_active_jobs():
        raise HTTPException(409, "Wait for audio generation to finish before unloading the model.")
    await asyncio.get_running_loop().run_in_executor(queue.executor, engine.unload)
    return engine.status()


@app.get("/api/voices")
def voices() -> dict[str, Any]:
    return {
        "predefined": storage.list_audio(storage.voices),
        "clone": storage.list_audio(storage.references),
    }


@app.post("/api/voices/upload")
async def upload_voice(
    kind: str = Query(..., pattern="^(predefined|clone)$"),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise HTTPException(400, "Unsupported audio format. Use WAV, MP3, FLAC, OGG, M4A or Opus.")
    directory = storage.voices if kind == "predefined" else storage.references
    filename = safe_filename(Path(file.filename or "voice").stem, "voice") + suffix
    destination = directory / filename
    total = 0
    with destination.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > 100 * 1024 * 1024:
                output.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(413, "Voice file exceeds 100 MB.")
            output.write(chunk)
    quality = await asyncio.to_thread(analyze_reference_voice, destination)
    return {"filename": filename, "size": total, "kind": kind, "quality": quality}


@app.get("/api/voices/{kind}/{filename}/quality")
async def voice_quality(kind: str, filename: str) -> dict[str, Any]:
    if kind not in {"predefined", "clone"}:
        raise HTTPException(400, "Invalid voice type.")
    try:
        path = storage.voice_path(kind, filename)
    except FileNotFoundError as error:
        raise HTTPException(404, "Voice file not found.") from error
    return await asyncio.to_thread(analyze_reference_voice, path)


@app.get("/api/voices/{kind}/{filename}")
def preview_voice(kind: str, filename: str, download: bool = False) -> FileResponse:
    if kind not in {"predefined", "clone"}:
        raise HTTPException(400, "Invalid voice type.")
    try:
        path = storage.voice_path(kind, filename)
    except FileNotFoundError as error:
        raise HTTPException(404, "Voice file not found.") from error
    return FileResponse(
        path,
        media_type=storage.media_type(path),
        filename=path.name if download else None,
        headers={"Cache-Control": "no-store", "Accept-Ranges": "bytes"},
    )


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return queue.list_jobs()


@app.post("/api/jobs")
async def create_job(request: AudioJobCreate) -> dict[str, Any]:
    try:
        return await queue.create(request)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/jobs/generate-all")
async def generate_all(request: GenerateAllRequest) -> list[dict[str, Any]]:
    try:
        return await queue.create_many(request.jobs)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    try:
        return queue.get(job_id)
    except KeyError as error:
        raise HTTPException(404, "Job not found.") from error


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, Any]:
    try:
        return await queue.cancel(job_id)
    except KeyError as error:
        raise HTTPException(404, "Job not found.") from error


@app.delete("/api/jobs")
async def remove_all_jobs(request: RemoveJobsRequest | None = None) -> dict[str, Any]:
    try:
        await queue.clear(delete_files=True if request is None else request.delete_files)
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from error
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
async def remove_job(job_id: str, delete_file: bool = True) -> dict[str, Any]:
    try:
        await queue.delete(job_id, delete_file=delete_file)
    except KeyError as error:
        raise HTTPException(404, "Job not found.") from error
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from error
    return {"ok": True}


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str, download: bool = False) -> FileResponse:
    try:
        job = queue.get(job_id)
    except KeyError as error:
        raise HTTPException(404, "Job not found.") from error
    if job["status"] != "completed" or not job["output_filename"]:
        raise HTTPException(409, "Audio is not ready.")
    path = storage.output_path(job["output_filename"])
    if not path.is_file():
        raise HTTPException(404, "Generated audio file is missing.")
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=path.name if download else None,
        headers={"Cache-Control": "no-store", "Accept-Ranges": "bytes"},
    )


@app.get("/api/jobs/{job_id}/asset/{kind}")
def job_asset(job_id: str, kind: str, download: bool = True) -> FileResponse:
    try:
        job = queue.get_raw(job_id)
    except KeyError as error:
        raise HTTPException(404, "Job not found.") from error
    fields = {
        "video-master": "video_master_filename",
        "srt": "srt_filename",
        "vtt": "vtt_filename",
    }
    field = fields.get(kind)
    if field is None:
        raise HTTPException(400, "Unknown output asset.")
    filename = job.get(field)
    if not filename:
        raise HTTPException(404, "This job does not have that output asset.")
    path = storage.output_path(filename)
    if not path.is_file():
        raise HTTPException(404, "Output asset not found.")
    return FileResponse(path, media_type=storage.media_type(path), filename=path.name if download else None)


def _waveform_peaks(path: Path, points: int) -> tuple[list[float], list[float], float]:
    cache = path.with_suffix(f".{points}.peaks.json")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        data = json.loads(cache.read_text(encoding="utf-8"))
        return data["mins"], data["maxs"], data["duration"]

    info = sf.info(path)
    block = max(1, int(info.frames / points))
    mins: list[float] = []
    maxs: list[float] = []
    with sf.SoundFile(path) as audio:
        while len(mins) < points:
            frames = audio.read(block, dtype="float32", always_2d=True)
            if not len(frames):
                break
            mono = frames.mean(axis=1)
            mins.append(float(np.min(mono)))
            maxs.append(float(np.max(mono)))
    data = {"mins": mins, "maxs": maxs, "duration": float(info.duration)}
    cache.write_text(json.dumps(data), encoding="utf-8")
    return mins, maxs, float(info.duration)


@app.get("/api/jobs/{job_id}/waveform")
async def waveform(job_id: str, points: int = Query(5000, ge=500, le=16000)) -> dict[str, Any]:
    try:
        job = queue.get(job_id)
    except KeyError as error:
        raise HTTPException(404, "Job not found.") from error
    if job["status"] != "completed" or not job["output_filename"]:
        raise HTTPException(409, "Audio is not ready.")
    path = storage.output_path(job["output_filename"])
    mins, maxs, duration = await asyncio.to_thread(_waveform_peaks, path, points)
    return {"mins": mins, "maxs": maxs, "duration": duration}


def _cut_audio(source: Path, destination: Path, start_seconds: float, end_seconds: float | None) -> float:
    with sf.SoundFile(source) as input_file:
        duration = float(len(input_file) / input_file.samplerate)
        start = min(max(start_seconds, 0.0), duration)
        end = duration if end_seconds is None else min(max(end_seconds, start), duration)
        input_file.seek(int(start * input_file.samplerate))
        remaining = int((end - start) * input_file.samplerate)
        with sf.SoundFile(
            destination,
            mode="w",
            samplerate=input_file.samplerate,
            channels=input_file.channels,
            subtype=input_file.subtype or "PCM_16",
        ) as output_file:
            while remaining > 0:
                frames = input_file.read(min(remaining, 65536), dtype="float32", always_2d=True)
                if not len(frames):
                    break
                output_file.write(frames)
                remaining -= len(frames)
    return end - start


@app.post("/api/jobs/{job_id}/cut")
async def cut_audio(job_id: str, request: CutRequest) -> dict[str, Any]:
    try:
        job = queue.get(job_id)
    except KeyError as error:
        raise HTTPException(404, "Job not found.") from error
    if job["status"] != "completed" or not job["output_filename"]:
        raise HTTPException(409, "Audio is not ready.")
    source = storage.output_path(job["output_filename"])
    title = safe_filename(job["title"] or f"Audio_{job['audio_number']}")
    prefix = safe_filename(request.filename_prefix, "Selected")
    filename = f"{prefix}_{title}.wav"
    destination = storage.output_path(filename)
    destination.unlink(missing_ok=True)
    duration = await asyncio.to_thread(
        _cut_audio,
        source,
        destination,
        request.start_seconds,
        request.end_seconds,
    )
    return {"filename": filename, "duration": duration, "url": f"/outputs/{filename}"}



@app.post("/tts")
async def tts(request: AudioJobCreate) -> FileResponse:
    job = await queue.create(request)
    job = await queue.wait(job["id"], timeout=60 * 60)
    if job["status"] != "completed":
        raise HTTPException(500, job.get("error") or "Generation failed.")
    path = storage.output_path(job["output_filename"])
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.post("/v1/audio/speech")
async def openai_speech(request: OpenAITTSRequest) -> FileResponse:
    valid_models = {model["id"] for model in MODELS}
    voice_mode = "predefined" if request.voice else "default"
    job_request = AudioJobCreate(
        audio_number=1,
        title="OpenAI_API",
        text=request.input,
        voice_mode=voice_mode,
        voice_filename=request.voice,
        options={
            "model": request.model if request.model in valid_models else "chatterbox",
            "speed_factor": request.speed,
        },
    )
    return await tts(job_request)


@app.post("/api/performance/feedback")
def save_performance_feedback(request: PerformanceFeedbackRequest) -> dict[str, Any]:
    try:
        job = queue.get_raw(request.job_id)
    except KeyError as error:
        raise HTTPException(404, "Job not found.") from error
    row = {
        **request.model_dump(),
        "model": job.get("options", {}).get("model"),
        "pace_profile": job.get("options", {}).get("senior_pace_profile"),
        "preset": job.get("preset"),
        "created_at": time.time(),
    }
    rows = storage.save_performance_feedback(row)
    matching = [item for item in rows if item.get("platform") == row["platform"] and item.get("model") == row["model"] and item.get("pace_profile") == row["pace_profile"]]
    avd = [float(item["average_view_duration_seconds"]) for item in matching if item.get("average_view_duration_seconds") is not None]
    intro = [float(item["intro_retention_percent"]) for item in matching if item.get("intro_retention_percent") is not None]
    return {
        "saved": True,
        "samples": len(matching),
        "average_view_duration_seconds": round(sum(avd) / len(avd), 1) if avd else None,
        "average_intro_retention_percent": round(sum(intro) / len(intro), 1) if intro else None,
    }


@app.get("/api/performance/summary")
def performance_summary() -> dict[str, Any]:
    rows = storage.load_performance_feedback()
    return {"samples": len(rows), "items": rows[-100:]}


@app.get("/v1/audio/voices")
def openai_voices() -> dict[str, Any]:
    return {"data": storage.list_audio(storage.voices) + storage.list_audio(storage.references)}
