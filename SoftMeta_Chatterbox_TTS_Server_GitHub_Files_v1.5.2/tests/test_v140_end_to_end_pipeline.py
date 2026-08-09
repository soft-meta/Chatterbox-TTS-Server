from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

# Self-contained engine import surface for queue tests without model weights.
stub = types.ModuleType("softmeta_chatterbox")
class _GenerationSettings:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
class _SoftMetaChatterboxEngine: pass
stub.GenerationSettings = _GenerationSettings
stub.SoftMetaChatterboxEngine = _SoftMetaChatterboxEngine
sys.modules.setdefault("softmeta_chatterbox", stub)

from models import AudioJobCreate, GenerationOptions
from queue_manager import QueueManager
from speech_pipeline import build_long_form_segments, prepare_senior_clear_speech_text
from emotion_director import is_heading


class _FakeStorage:
    def __init__(self, root: Path):
        self.root = root
        self.saved = {}
    def load_jobs(self): return {}
    def save_jobs(self, jobs): self.saved = dict(jobs)
    def output_path(self, filename): return self.root / filename
    def voice_path(self, mode, filename): return None


class _FakeFastEngine:
    def generate(self, text, *, model_name, reference_audio, language, options):
        words = max(1, len([w for w in text.replace("[narration]", "").replace("[happy]", "").replace("[surprised]", "").replace("[dramatic]", "").split() if w]))
        sr = 24000
        seconds = words / 180.0 * 60.0  # deliberately too fast; adaptive pacing should slow it.
        n = max(1, int(sr * seconds))
        t = np.arange(n, dtype=np.float32) / sr
        audio = (0.07 * np.sin(2 * np.pi * 190 * t) + 0.012 * np.sin(2 * np.pi * 2400 * t)).astype(np.float32)
        return SimpleNamespace(waveform=torch.from_numpy(audio).unsqueeze(0), sample_rate=sr)


def _measure_lufs(path: Path) -> float:
    proc = subprocess.run([
        "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info", "-i", str(path),
        "-af", "loudnorm=I=-12.5:TP=-1.0:LRA=4.5:print_format=json", "-f", "null", "-",
    ], check=True, capture_output=True, text=True)
    start, end = proc.stderr.rfind("{"), proc.stderr.rfind("}")
    values = json.loads(proc.stderr[start:end + 1])
    return float(values["input_i"])


def test_clear_speech_preserves_heading_for_section_director():
    text = "Number One: Protect Your Sleep\nRemember this because a steady bedtime can make tomorrow easier."
    prepared = prepare_senior_clear_speech_text(text)
    assert "Number One: Protect Your Sleep" in prepared
    segments = build_long_form_segments(prepared, max_words=85, heading_detector=is_heading)
    assert segments[0].text.startswith("Number One. Protect Your Sleep.")


def test_mock_longform_queue_executes_professional_pipeline(tmp_path: Path):
    async def run():
        storage = _FakeStorage(tmp_path)
        manager = QueueManager(engine=_FakeFastEngine(), storage=storage)
        text = """Why Daily Habits Matter After 70

A good routine should feel calm and practical. Most people do not realize how easy it is to overlook small changes when every day feels similar. Remember this because the goal is not to rush. The goal is to notice, understand, and make one useful change at a time.

Number One: Protect Your Morning Routine
A steady morning can make the rest of the day easier. Keep the first hour simple, drink fluids as your doctor allows, and give yourself enough time to move without rushing. The important thing is to build a routine you can actually repeat.

Number Two: Pay Attention to New Warning Signs
Do not ignore sudden chest pressure or severe shortness of breath. New symptoms deserve calm attention, and serious symptoms may need urgent medical help. Keep a short note of what changed so you can explain it clearly.

Number Three: Keep the Plan Realistic
The good news is that small changes can help. You do not need a perfect routine. You need a routine that is clear, safe, and easy enough to continue tomorrow."""
        request = AudioJobCreate(
            preset="Motivational Speech",
            audio_number=1,
            title="Pipeline Test",
            text=text,
            voice_mode="default",
            options=GenerationOptions(model="chatterbox-turbo", split_text=True, chunk_words=85),
        )
        public = await manager.create(request, enqueue=False)
        job = manager.get_raw(public["id"])
        await manager._process(job)
        path = tmp_path / job["output_filename"]
        assert path.exists()
        info = sf.info(path)
        # Output should land near senior-friendly long-form pace, not the fake 180 WPM input.
        effective_wpm = job["total_words"] / (info.duration / 60.0)
        assert 138 <= effective_wpm <= 164, effective_wpm
        assert abs(_measure_lufs(path) - (-12.5)) < 0.6
        assert job["status"] == "completed"
        await manager.stop()
    asyncio.run(run())
