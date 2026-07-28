from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import ROOT, load_config
from engine import EngineService
from models import (
    AudioJobCreate,
    CutRequest,
    GenerateAllRequest,
    ModelLoadRequest,
    OpenAITTSRequest,
)
from queue_manager import QueueManager
from storage import AUDIO_EXTENSIONS, Storage
from utils import safe_filename

config = load_config()
storage = Storage()
engine = EngineService(device=config["tts_engine"]["device"])
queue = QueueManager(engine=engine, storage=storage)

PRESETS = [
    {
        "name": "Motivational Speech",
        "description": "Natural US-English motivational narration with controlled emotion.",
        "language": "en",
        "temperature": 0.8,
        "exaggeration": 0.65,
        "cfg_weight": 0.35,
        "repetition_penalty": 1.2,
        "min_p": 0.05,
        "top_p": 1.0,
        "top_k": 1000,
        "speed_factor": 1.0,
        "seed": 2025,
        "split_text": True,
        "chunk_words": 90,
    },
    {
        "name": "Natural Conversation",
        "description": "Balanced, calm delivery for general narration.",
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
    },
]


@asynccontextmanager
async def lifespan(_: FastAPI):
    await queue.start()
    yield
    await queue.stop()


app = FastAPI(
    title="Soft Meta Chatterbox TTS Server",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=ROOT / "ui"), name="static")
app.mount("/outputs", StaticFiles(directory=storage.outputs), name="outputs")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((ROOT / "ui" / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "device": engine.device,
        "loaded_model": engine.loaded_model,
        "queue_size": queue.queue.qsize(),
    }


@app.get("/api/ui/initial-data")
def initial_data() -> dict[str, Any]:
    return {
        "app": {"name": "Soft Meta Chatterbox TTS Server", "version": "0.1.0"},
        "models": [
            {"id": "chatterbox", "name": "Chatterbox Original (English)"},
            {"id": "chatterbox-turbo", "name": "Chatterbox Turbo (English)"},
            {"id": "chatterbox-nano", "name": "Chatterbox Nano (English)"},
            {"id": "chatterbox-multilingual", "name": "Chatterbox Multilingual V3"},
        ],
        "active_model": engine.loaded_model or config["tts_engine"]["default_model"],
        "model_loaded": engine.loaded_model is not None,
        "device": engine.device,
        "predefined_voices": storage.list_audio(storage.voices),
        "reference_voices": storage.list_audio(storage.references),
        "presets": PRESETS,
        "defaults": config["generation_defaults"],
        "jobs": queue.list_jobs(),
    }


@app.post("/api/model/load")
async def load_model(request: ModelLoadRequest) -> dict[str, Any]:
    if any(job["status"] == "running" for job in queue.jobs.values()):
        raise HTTPException(409, "Wait for the current audio generation to finish before changing models.")
    await asyncio.get_running_loop().run_in_executor(queue.executor, engine.load, request.model)
    return {"loaded_model": engine.loaded_model, "device": engine.device}


@app.get("/api/voices")
def voices() -> dict[str, Any]:
    return {
        "predefined": storage.list_audio(storage.voices),
        "clone": storage.list_audio(storage.references),
    }


@app.post("/api/voices/upload")
async def upload_voice(kind: str = Query(..., pattern="^(predefined|clone)$"), file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise HTTPException(400, "Unsupported audio format.")
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
    return {"filename": filename, "size": total, "kind": kind}


@app.get("/api/voices/{kind}/{filename}")
def preview_voice(kind: str, filename: str) -> FileResponse:
    if kind not in {"predefined", "clone"}:
        raise HTTPException(400, "Invalid voice type.")
    try:
        path = storage.voice_path(kind, filename)
    except FileNotFoundError as error:
        raise HTTPException(404, "Voice file not found.") from error
    return FileResponse(path, media_type=storage.media_type(path), filename=path.name)


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return queue.list_jobs()


@app.post("/api/jobs")
async def create_job(request: AudioJobCreate) -> dict[str, Any]:
    return await queue.create(request)


@app.post("/api/jobs/generate-all")
async def generate_all(request: GenerateAllRequest) -> list[dict[str, Any]]:
    return await queue.create_many(request.jobs)


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


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str) -> FileResponse:
    try:
        job = queue.get(job_id)
    except KeyError as error:
        raise HTTPException(404, "Job not found.") from error
    if job["status"] != "completed" or not job["output_filename"]:
        raise HTTPException(409, "Audio is not ready.")
    path = storage.output_path(job["output_filename"])
    if not path.is_file():
        raise HTTPException(404, "Generated audio file is missing.")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


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
async def waveform(job_id: str, points: int = Query(4000, ge=500, le=12000)) -> dict[str, Any]:
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
    filename = f"{prefix}_{title}_{int(time.time())}.wav"
    destination = storage.output_path(filename)
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
    voice_mode = "predefined" if request.voice else "default"
    job_request = AudioJobCreate(
        audio_number=1,
        title="OpenAI_API",
        text=request.input,
        voice_mode=voice_mode,
        voice_filename=request.voice,
        options={"model": request.model if request.model in {
            "chatterbox", "chatterbox-turbo", "chatterbox-nano", "chatterbox-multilingual"
        } else "chatterbox", "speed_factor": request.speed},
    )
    return await tts(job_request)


@app.get("/v1/audio/voices")
def openai_voices() -> dict[str, Any]:
    return {"data": storage.list_audio(storage.voices) + storage.list_audio(storage.references)}
