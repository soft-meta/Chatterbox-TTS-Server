from __future__ import annotations

import numpy as np

from config import DEFAULT_CONFIG
from models import GenerationOptions
from quality_control import QualityController


def _audio(words: int = 40, wpm: float = 150.0, sr: int = 24000) -> np.ndarray:
    seconds = words / wpm * 60.0
    t = np.arange(max(1, int(sr * seconds)), dtype=np.float32) / sr
    return (0.06 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def test_turbo_is_default_while_original_remains_available() -> None:
    assert DEFAULT_CONFIG["tts_engine"]["default_model"] == "chatterbox-turbo"
    assert GenerationOptions().model == "chatterbox-turbo"


def test_asr_repetition_and_35_percent_mismatch_are_advisory_not_hard_failure() -> None:
    expected = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    # Deliberately bad/repetitive verifier transcript. Audio itself remains acoustically healthy.
    transcript = "one two three four five one two three four five one two three four five six seven"
    qc = QualityController()
    qc.transcribe = lambda *_args, **_kwargs: (transcript, [], True)  # type: ignore[method-assign]
    qc.speaker_similarity = lambda *_args, **_kwargs: (None, False)  # type: ignore[method-assign]
    report = qc.evaluate(_audio(20), 24000, expected, None, [])
    assert report.passed is True
    assert report.retry_recommended is True
    assert report.hard_failure is False
    assert any("repeated" in reason or "ASR" in reason for reason in report.reasons)


def test_even_extreme_asr_mismatch_does_not_abort_healthy_audio() -> None:
    expected = " ".join(f"word{i}" for i in range(50))
    qc = QualityController()
    qc.transcribe = lambda *_args, **_kwargs: ("completely different verifier text", [], True)  # type: ignore[method-assign]
    qc.speaker_similarity = lambda *_args, **_kwargs: (None, False)  # type: ignore[method-assign]
    report = qc.evaluate(_audio(50), 24000, expected, None, [])
    assert report.passed is True
    assert report.retry_recommended is True
    assert report.hard_failure is False


def test_objectively_unusable_audio_still_hard_fails() -> None:
    qc = QualityController()
    qc.transcribe = lambda *_args, **_kwargs: ("", [], False)  # type: ignore[method-assign]
    qc.speaker_similarity = lambda *_args, **_kwargs: (None, False)  # type: ignore[method-assign]
    report = qc.evaluate(np.zeros(24000 * 3, dtype=np.float32), 24000, "this should contain real speech", None, [])
    assert report.passed is False
    assert report.hard_failure is True



def test_original_long_job_completes_with_repeated_asr_warnings(tmp_path) -> None:
    import asyncio
    import sys
    import types
    from types import SimpleNamespace
    from pathlib import Path

    # QueueManager imports the engine adapter in normal runtime; tests use a tiny stub.
    if "softmeta_chatterbox" not in sys.modules:
        stub = types.ModuleType("softmeta_chatterbox")
        class _GenerationSettings:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        class _SoftMetaChatterboxEngine: pass
        stub.GenerationSettings = _GenerationSettings
        stub.SoftMetaChatterboxEngine = _SoftMetaChatterboxEngine
        sys.modules["softmeta_chatterbox"] = stub

    from models import AudioJobCreate, GenerationOptions
    from quality_control import ChunkQualityReport
    from queue_manager import QueueManager
    from utils import count_words

    class Storage:
        def __init__(self, root: Path): self.root = root
        def load_jobs(self): return {}
        def save_jobs(self, jobs): pass
        def output_path(self, name: str): return self.root / name
        def resolve_voice(self, *args, **kwargs): return None

    class Engine:
        def __init__(self): self.calls = 0; self.models = []
        def generate(self, text, model_name, reference_audio, language, options):
            self.calls += 1; self.models.append(model_name)
            sr = 24000
            words = max(1, count_words(text))
            seconds = words / 150.0 * 60.0
            t = np.arange(max(1, int(sr * seconds)), dtype=np.float32) / sr
            wav = (0.05 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
            return SimpleNamespace(
                waveform=SimpleNamespace(squeeze=lambda: SimpleNamespace(numpy=lambda: wav)),
                sample_rate=sr,
            )

    class AdvisoryQuality:
        def evaluate(self, audio, sample_rate, expected_text, reference_path, speaker_history):
            return ChunkQualityReport(
                passed=True, score=48.0,
                reasons=["high ASR mismatch (35%)", "possible repeated or hallucinated words"],
                transcript="imperfect repeated verifier transcript", wer=0.35,
                metrics={"wpm": 150.0}, asr_available=True,
                retry_recommended=True, hard_failure=False,
            )

    async def run():
        engine = Engine()
        manager = QueueManager(engine=engine, storage=Storage(Path(tmp_path)))
        manager.quality = AdvisoryQuality()
        text = " ".join(f"sentenceword{i}" for i in range(280)) + "."
        req = AudioJobCreate(
            preset="Motivational Speech", audio_number=1, title="original advisory",
            text=text, voice_mode="default",
            options=GenerationOptions(model="chatterbox", split_text=True, platform_assets=False),
        )
        public = await manager.create(req, enqueue=False)
        job = manager.get_raw(public["id"])
        await manager._process(job)
        assert job["status"] == "completed"
        assert set(engine.models) == {"chatterbox"}
        assert job["quality_summary"]["hard_failures"] == 0
        assert job["quality_summary"]["warning_chunks"] >= 1
        await manager.stop()

    asyncio.run(run())
