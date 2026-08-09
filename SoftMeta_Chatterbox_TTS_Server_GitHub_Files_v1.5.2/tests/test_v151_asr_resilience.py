from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

# QueueManager imports the separately installed SoftMeta engine adapter.
stub = types.ModuleType("softmeta_chatterbox")
class _GenerationSettings:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
class _SoftMetaChatterboxEngine: pass
stub.GenerationSettings = _GenerationSettings
stub.SoftMetaChatterboxEngine = _SoftMetaChatterboxEngine
sys.modules.setdefault("softmeta_chatterbox", stub)

from models import AudioJobCreate, GenerationOptions
from quality_control import ChunkQualityReport, QualityController
from queue_manager import QueueManager


def _senior_audio(words: int, sr: int = 24000, wpm: float = 150.0) -> np.ndarray:
    seconds = words / wpm * 60.0
    t = np.arange(max(1, int(sr * seconds)), dtype=np.float32) / sr
    return (0.08 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def test_35_percent_asr_mismatch_is_retry_warning_not_hard_failure() -> None:
    expected = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    # 13/20 words recovered in order: intentionally imperfect ASR, but audio length
    # is normal for the expected senior-paced speech.
    transcript = "one two three four five seven nine ten twelve fourteen sixteen eighteen twenty"
    qc = QualityController()
    qc.transcribe = lambda *_args, **_kwargs: (transcript, [], True)  # type: ignore[method-assign]
    qc.speaker_similarity = lambda *_args, **_kwargs: (None, False)  # type: ignore[method-assign]
    report = qc.evaluate(_senior_audio(20), 24000, expected, None, [])
    assert report.wer is not None and 0.30 < report.wer < 0.50
    assert report.passed is True
    assert report.retry_recommended is True
    assert report.hard_failure is False


def test_corroborated_missing_speech_still_hard_fails() -> None:
    expected = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    transcript = "one two three four"
    qc = QualityController()
    qc.transcribe = lambda *_args, **_kwargs: (transcript, [], True)  # type: ignore[method-assign]
    qc.speaker_similarity = lambda *_args, **_kwargs: (None, False)  # type: ignore[method-assign]
    # Much too short for 20 expected words, corroborating the ASR evidence.
    report = qc.evaluate(_senior_audio(8, wpm=180.0), 24000, expected, None, [])
    assert report.passed is False
    assert report.hard_failure is True


class _Storage:
    def __init__(self, root: Path): self.root = root
    def load_jobs(self): return {}
    def save_jobs(self, jobs): pass
    def output_path(self, name: str) -> Path: return self.root / name
    def resolve_voice(self, *args, **kwargs): return None


class _Engine:
    def __init__(self): self.calls = 0
    def generate(self, text, model_name, reference_audio, language, options):
        self.calls += 1
        sr = 24000
        words = max(1, len(text.replace('[narration]', '').split()))
        return SimpleNamespace(waveform=SimpleNamespace(squeeze=lambda: SimpleNamespace(numpy=lambda: _senior_audio(words, sr))), sample_rate=sr)


class _SoftWarningQuality:
    def evaluate(self, audio, sample_rate, expected_text, reference_path, speaker_history):
        return ChunkQualityReport(
            passed=True, score=67.0, reasons=["high ASR mismatch (35%)", "ASR heard fewer words than expected"],
            transcript="imperfect verifier transcript", wer=0.35, speaker_similarity=None,
            metrics={"wpm": 150.0}, words=[], asr_available=True, speaker_check_available=False,
            retry_recommended=True, hard_failure=False,
        )


def test_soft_asr_warning_retries_but_does_not_abort_long_job(tmp_path: Path) -> None:
    async def run() -> None:
        engine = _Engine()
        manager = QueueManager(engine=engine, storage=_Storage(tmp_path))
        manager.quality = _SoftWarningQuality()
        request = AudioJobCreate(
            preset="Motivational Speech", audio_number=1, title="ASR resilience",
            text="This is a calm complete sentence for an older listener and it should not fail only because the ASR verifier is uncertain.",
            voice_mode="default", options=GenerationOptions(model="chatterbox-turbo", split_text=True, platform_assets=False),
        )
        public = await manager.create(request, enqueue=False)
        job = manager.get_raw(public["id"])
        await manager._process(job)
        assert job["status"] == "completed"
        assert engine.calls == 3  # initial + two focused retries
        assert job["quality_summary"]["warning_chunks"] >= 1
        assert list(tmp_path.glob("Audio_*.wav"))
        await manager.stop()
    asyncio.run(run())
