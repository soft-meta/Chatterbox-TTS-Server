from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# queue_manager imports the separately installed adapter; use a tiny unit-test stub.
stub = types.ModuleType("softmeta_chatterbox")
class _GenerationSettings:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
class _SoftMetaChatterboxEngine: pass
stub.GenerationSettings = _GenerationSettings
stub.SoftMetaChatterboxEngine = _SoftMetaChatterboxEngine
sys.modules.setdefault("softmeta_chatterbox", stub)

from models import AudioJobCreate, GenerationOptions
from queue_manager import QueueManager, OFFICIAL_TURBO_STANDARD_PROFILE


class _Storage:
    def __init__(self, root: Path): self.root = root
    def load_jobs(self): return {}
    def save_jobs(self, _jobs): pass
    def output_path(self, filename): return self.root / filename
    def voice_path(self, _mode, _filename): return None


class _CaptureEngine:
    def __init__(self): self.calls: list[tuple[str, str, dict]] = []
    def generate(self, text, *, model_name, reference_audio, language, options):
        self.calls.append((model_name, text, dict(options)))
        sr = 24000
        words = max(1, len(text.replace("[", " ").replace("]", " ").split()))
        seconds = max(0.8, words / 148.0 * 60.0)
        t = np.arange(int(sr * seconds), dtype=np.float32) / sr
        audio = (0.06 * np.sin(2 * np.pi * 185 * t)).astype(np.float32)
        return SimpleNamespace(waveform=torch.from_numpy(audio).unsqueeze(0), sample_rate=sr)


def _request(text: str, mode: str) -> AudioJobCreate:
    return AudioJobCreate(
        preset="Motivational Speech",
        generation_mode=mode,
        auto_emotion=(mode == "advanced"),
        audio_number=1,
        title=f"{mode} comparison",
        text=text,
        voice_mode="default",
        options=GenerationOptions(
            model="chatterbox-turbo",
            split_text=True,
            chunk_words=85,
            quality_gate=False,
            speaker_consistency=False,
            platform_assets=False,
        ),
    )


def test_standard_profile_matches_official_turbo_demo_defaults_and_disables_softmeta_pipeline() -> None:
    manager = QueueManager(engine=SimpleNamespace(), storage=SimpleNamespace(load_jobs=lambda: {}, save_jobs=lambda _jobs: None))
    try:
        options = manager._effective_options(_request("A short clean comparison sentence.", "standard"))
        for key, expected in OFFICIAL_TURBO_STANDARD_PROFILE.items():
            assert options[key] == expected
        assert options["temperature"] == 0.80
        assert options["top_p"] == 0.95
        assert options["top_k"] == 1000
        assert options["repetition_penalty"] == 1.20
        assert options["min_p"] == 0.0
        assert options["split_text"] is False
        assert options["quality_gate"] is False
        assert options["platform_assets"] is False
        assert options["seed"] > 0
    finally:
        manager.executor.shutdown(wait=False, cancel_futures=True)


def test_standard_turbo_is_one_direct_call_and_skips_softmeta_mastering(tmp_path: Path, monkeypatch) -> None:
    engine = _CaptureEngine()
    manager = QueueManager(engine=engine, storage=_Storage(tmp_path))
    # Any call to these would prove the baseline is contaminated by the advanced path.
    monkeypatch.setattr(manager, "_normalize_turbo_loudness", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("standard must not post-normalize")))
    monkeypatch.setattr("queue_manager.master_professional_voice", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("standard must not master")))
    text = "This is my untouched baseline. [happy] This user text stays untouched. [laugh] This manual official event also stays untouched."

    async def run():
        public = await manager.create(_request(text, "standard"), enqueue=False)
        job = manager.get_raw(public["id"])
        assert job["emotion_summary"] is None
        assert job["generation_text"] == text
        assert "[happy]" in job["generation_text"]
        assert "[laugh]" in job["generation_text"]
        await manager._process(job)
        await manager.stop()
        return job

    job = asyncio.run(run())
    assert job["status"] == "completed"
    assert len(engine.calls) == 1
    model, sent_text, sent_options = engine.calls[0]
    assert model == "chatterbox-turbo"
    assert sent_text == text
    assert "[happy]" in sent_text
    assert "[laugh]" in sent_text
    assert sent_options["temperature"] == 0.80
    assert sent_options["top_p"] == 0.95
    assert sent_options["top_k"] == 1000


def test_advanced_auto_events_reach_turbo_engine_calls(tmp_path: Path, monkeypatch) -> None:
    engine = _CaptureEngine()
    manager = QueueManager(engine=engine, storage=_Storage(tmp_path))
    monkeypatch.setattr("queue_manager.apply_tempo_array", lambda audio, _sr, _factor: audio)
    monkeypatch.setattr("queue_manager.master_professional_voice", lambda _path, **_kwargs: {"target_lufs": -13.0})
    monkeypatch.setattr("queue_manager.analyze_prosody_quality", lambda *_a, **_k: {"score": 90.0, "rating": "Strong"})

    paragraph = (
        "My grandson told me I seemed happier, and I laughed because he was right. "
        "That small moment still makes me smile when I remember it. "
        "But I regret the years I spent carrying every problem alone, and I wish I had known sooner. "
        "A few weeks later I saw a result I never expected, and it stopped me in my tracks. "
        "I wrote the lesson down so I would remember it and share it calmly with other people. "
    )
    text = " ".join([paragraph] * 6)

    async def run():
        public = await manager.create(_request(text, "advanced"), enqueue=False)
        job = manager.get_raw(public["id"])
        tagged = job["generation_text"].lower()
        assert any(tag in tagged for tag in ("[laugh]", "[chuckle]"))
        assert "[sigh]" in tagged
        assert "[gasp]" in tagged
        assert job["emotion_summary"]["applied_count"] >= 3
        await manager._process(job)
        await manager.stop()
        return job

    job = asyncio.run(run())
    assert job["status"] == "completed"
    sent = "\n".join(text for _model, text, _options in engine.calls).lower()
    assert any(tag in sent for tag in ("[laugh]", "[chuckle]")), sent
    assert "[sigh]" in sent, sent
    assert "[gasp]" in sent, sent
    assert len(engine.calls) > 1  # advanced path is segmented/directed


def test_ui_exposes_single_professional_generate_button_and_opt_in_auto_emotion() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "ui" / "index.html").read_text(encoding="utf-8")
    js = (root / "ui" / "app.js").read_text(encoding="utf-8")
    assert 'data-action="generate-audio"' in html
    assert 'data-action="generate-standard"' not in html
    assert 'data-action="generate-advanced"' not in html
    assert 'Generate Advanced Audio' not in html
    assert 'data-action="toggle-auto-emotion"' in html
    assert "generateOne(tab)" in js
    assert "generation_mode: 'advanced'" in js
    assert "auto_emotion: Boolean(tab.auto_emotion)" in js


def test_original_generate_audio_request_is_coerced_to_existing_advanced_path(tmp_path: Path) -> None:
    manager = QueueManager(engine=SimpleNamespace(), storage=_Storage(tmp_path))
    request = AudioJobCreate(
        preset="Motivational Speech",
        generation_mode="standard",
        audio_number=1,
        title="original frozen",
        text="The good news is that simple routines can help. I learned too late that rushing made things harder. " * 4,
        voice_mode="default",
        options=GenerationOptions(model="chatterbox", platform_assets=False),
    )
    async def run():
        public = await manager.create(request, enqueue=False)
        job = manager.get_raw(public["id"])
        await manager.stop()
        return job
    job = asyncio.run(run())
    assert job["generation_mode"] == "advanced"
    assert job["emotion_summary"] is not None
