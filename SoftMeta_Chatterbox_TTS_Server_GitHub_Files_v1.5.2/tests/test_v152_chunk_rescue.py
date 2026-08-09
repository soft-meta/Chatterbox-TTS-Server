from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

stub = types.ModuleType("softmeta_chatterbox")
class _GenerationSettings:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
class _SoftMetaChatterboxEngine: pass
stub.GenerationSettings = _GenerationSettings
stub.SoftMetaChatterboxEngine = _SoftMetaChatterboxEngine
sys.modules.setdefault("softmeta_chatterbox", stub)

from models import AudioJobCreate, GenerationOptions
from quality_control import ChunkQualityReport
from queue_manager import QueueManager
from utils import count_words


class _Storage:
    def __init__(self, root: Path): self.root = root
    def load_jobs(self): return {}
    def save_jobs(self, jobs): pass
    def output_path(self, name: str) -> Path: return self.root / name
    def resolve_voice(self, *args, **kwargs): return None
    def clear_outputs(self): pass
    def delete_output_artifacts(self, *args, **kwargs): pass


class _Engine:
    def __init__(self): self.calls: list[int] = []
    def generate(self, text, model_name, reference_audio, language, options):
        words = max(1, count_words(text))
        self.calls.append(words)
        sr = 24000
        seconds = words / 150.0 * 60.0
        t = np.arange(max(1, int(sr * seconds)), dtype=np.float32) / sr
        wav = (0.05 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        return SimpleNamespace(
            waveform=SimpleNamespace(squeeze=lambda: SimpleNamespace(numpy=lambda: wav)),
            sample_rate=sr,
        )


class _LengthSensitiveQuality:
    """Reproduce the user's failure: long span fails all retries, smaller span is clean."""
    def evaluate(self, audio, sample_rate, expected_text, reference_path, speaker_history):
        words = count_words(expected_text)
        if words > 40:
            return ChunkQualityReport(
                passed=False,
                score=20.0,
                reasons=[
                    "abnormal speaking rate",
                    "high ASR mismatch (65%)",
                    "ASR heard fewer words than expected",
                ],
                transcript="incomplete verifier transcript",
                wer=0.65,
                metrics={"wpm": 70.0},
                asr_available=True,
                retry_recommended=True,
                hard_failure=True,
            )
        return ChunkQualityReport(
            passed=True,
            score=96.0,
            reasons=[],
            transcript=expected_text,
            wer=0.02,
            metrics={"wpm": 150.0},
            asr_available=True,
            retry_recommended=False,
            hard_failure=False,
        )


def test_hard_failed_long_chunk_is_rescued_by_verified_smaller_subchunks(tmp_path: Path) -> None:
    async def run() -> None:
        engine = _Engine()
        manager = QueueManager(engine=engine, storage=_Storage(tmp_path))
        manager.quality = _LengthSensitiveQuality()
        text = " ".join(f"word{i}" for i in range(170)) + "."
        request = AudioJobCreate(
            preset="Motivational Speech",
            audio_number=1,
            title="rescue",
            text=text,
            voice_mode="default",
            options=GenerationOptions(
                model="chatterbox-turbo",
                split_text=True,
                platform_assets=False,
            ),
        )
        public = await manager.create(request, enqueue=False)
        job = manager.get_raw(public["id"])
        await manager._process(job)
        assert job["status"] == "completed"
        # The first 72-word span fails initial + 2 ordinary retries, then the rescue
        # path switches to materially smaller spans that pass QC.
        assert engine.calls[:3] == [72, 72, 72]
        assert any(words <= 36 for words in engine.calls[3:])
        assert max(engine.calls[3:6]) <= 36
        assert job["quality_summary"]["hard_failures"] == 0
        assert any(
            "recovered with" in warning
            for warning in job["quality_summary"]["warnings"]
        )
        await manager.stop()
    asyncio.run(run())


def test_rescue_does_not_weaken_hard_failure_if_small_segments_also_fail(tmp_path: Path) -> None:
    class _AlwaysBadQuality:
        def evaluate(self, audio, sample_rate, expected_text, reference_path, speaker_history):
            return ChunkQualityReport(
                passed=False,
                score=5.0,
                reasons=["strong evidence of missing speech"],
                transcript="",
                wer=0.95,
                metrics={"wpm": 55.0},
                asr_available=True,
                retry_recommended=True,
                hard_failure=True,
            )

    async def run() -> None:
        manager = QueueManager(engine=_Engine(), storage=_Storage(tmp_path))
        manager.quality = _AlwaysBadQuality()
        request = AudioJobCreate(
            preset="Motivational Speech", audio_number=1, title="still bad",
            text=" ".join(f"word{i}" for i in range(90)) + ".",
            voice_mode="default",
            options=GenerationOptions(model="chatterbox-turbo", split_text=True, platform_assets=False),
        )
        public = await manager.create(request, enqueue=False)
        job = manager.get_raw(public["id"])
        try:
            await manager._process(job)
        except RuntimeError as error:
            assert "Production Quality Gate rejected chunk" in str(error)
        else:
            raise AssertionError("A genuinely bad small-segment rescue must still hard-fail")
        await manager.stop()
    asyncio.run(run())


def test_original_uses_the_same_verified_rescue_pipeline(tmp_path: Path) -> None:
    async def run() -> None:
        engine = _Engine()
        manager = QueueManager(engine=engine, storage=_Storage(tmp_path))
        manager.quality = _LengthSensitiveQuality()
        request = AudioJobCreate(
            preset="Motivational Speech", audio_number=1, title="original rescue",
            text=" ".join(f"word{i}" for i in range(100)) + ".",
            voice_mode="default",
            options=GenerationOptions(model="chatterbox", split_text=True, platform_assets=False),
        )
        public = await manager.create(request, enqueue=False)
        job = manager.get_raw(public["id"])
        await manager._process(job)
        assert job["status"] == "completed"
        assert any(words <= 36 for words in engine.calls)
        assert job["quality_summary"]["hard_failures"] == 0
        await manager.stop()
    asyncio.run(run())
