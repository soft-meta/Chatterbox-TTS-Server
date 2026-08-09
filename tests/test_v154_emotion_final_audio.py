from __future__ import annotations

import asyncio
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# QueueManager imports the separately installed adapter; unit tests use a light stub.
stub = types.ModuleType("softmeta_chatterbox")
class _GenerationSettings:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
class _SoftMetaChatterboxEngine: pass
stub.GenerationSettings = _GenerationSettings
stub.SoftMetaChatterboxEngine = _SoftMetaChatterboxEngine
sys.modules.setdefault("softmeta_chatterbox", stub)

from emotion_director import analyze_serious_senior_advisor, is_heading
from models import AudioJobCreate, GenerationOptions
from quality_control import ChunkQualityReport
from queue_manager import QueueManager
from speech_pipeline import build_long_form_segments


def _rms_db(audio: np.ndarray) -> float:
    value = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))
    return 20.0 * math.log10(max(value, 1e-9))


def test_longform_planner_isolates_emotion_sentence_in_compact_segment() -> None:
    text = (
        "A calm opening sentence explains the topic and gives enough context for the listener. "
        "Another neutral sentence keeps the introduction steady and easy to follow. "
        "[happy] The good news is that a simple daily habit can make this easier to manage. "
        "The next neutral sentence should return to the normal advisor delivery without carrying the happy control. "
        "[dramatic] Do not ignore sudden chest pressure or severe shortness of breath when it appears. "
        "The closing sentence returns to calm practical advice for the viewer."
    )
    segments = build_long_form_segments(text, max_words=85, heading_detector=is_heading, age_profile="70s")
    expressive = [segment for segment in segments if segment.emotion_tag]
    assert [segment.emotion_tag for segment in expressive] == ["happy", "dramatic"]
    assert all(segment.word_count <= 44 for segment in expressive)
    assert all(segment.role == "emotion" for segment in expressive)
    assert "[happy]" in expressive[0].text
    assert "[dramatic]" in expressive[1].text
    assert any(segment.emotion_tag is None for segment in segments)


def test_original_emotion_survives_conservative_retry() -> None:
    class _Storage:
        def load_jobs(self): return {}
        def save_jobs(self, _jobs): pass
    manager = QueueManager(engine=SimpleNamespace(), storage=_Storage())
    try:
        neutral_retry = {"temperature": 0.72, "exaggeration": 0.58, "cfg_weight": 0.35, "_quality_retry": True}
        spoken, options = manager._original_chunk_direction(
            "[surprised] Most people do not realize how much this small change can matter.",
            neutral_retry,
            "surprised",
        )
        assert "[surprised]" not in spoken
        assert options["_emotion_tag"] == "surprised"
        # v1.5.3 reset retry exaggeration to 0.52 and erased the expression. The
        # retry now stays conservative but retains a clearly different emotion profile.
        assert options["exaggeration"] >= 0.66
        assert options["cfg_weight"] <= 0.30
        assert options["temperature"] < 0.60
    finally:
        manager.executor.shutdown(wait=False, cancel_futures=True)


def test_professional_chunk_leveler_preserves_intentional_emotion_energy() -> None:
    sr = 24000
    t = np.arange(sr, dtype=np.float32) / sr
    neutral = (0.18 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    expressive = (0.25 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    neutral_out = QueueManager._level_professional_chunk(neutral, None)
    expressive_out = QueueManager._level_professional_chunk(expressive, "happy")
    # Both are already inside safe professional bands, so their natural energy
    # difference should survive instead of being forced to identical RMS.
    difference = _rms_db(expressive_out) - _rms_db(neutral_out)
    assert difference > 2.0
    assert float(np.max(np.abs(expressive_out))) < 0.90


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
        words = max(1, len(text.split()))
        seconds = max(1.2, words / 148.0 * 60.0)
        t = np.arange(int(sr * seconds), dtype=np.float32) / sr
        amp = 0.075 if options.get("_emotion_tag") else 0.065
        audio = (amp * np.sin(2 * np.pi * 185 * t)).astype(np.float32)
        return SimpleNamespace(waveform=torch.from_numpy(audio).unsqueeze(0), sample_rate=sr)


class _AsrWarningQuality:
    def evaluate(self, audio, sample_rate, expected_text, reference_path, speaker_history):
        # Simulate the exact advisory class on expressive beats only. Neutral chunks
        # pass cleanly so this test proves ASR uncertainty cannot flatten emotion by
        # repeatedly resampling the directed segment.
        lower = expected_text.lower()
        expressive = any(phrase in lower for phrase in ("good news", "do not ignore", "not always the whole story"))
        return ChunkQualityReport(
            passed=True,
            score=55.0 if expressive else 100.0,
            reasons=["high ASR mismatch (35%)"] if expressive else [],
            transcript=expected_text,
            wer=0.35 if expressive else 0.0,
            speaker_similarity=None,
            metrics={"wpm": 148.0},
            words=[],
            asr_available=True,
            speaker_check_available=False,
            retry_recommended=expressive,
            hard_failure=False,
        )


def _emotion_script() -> str:
    return (
        "Many older adults hear a lot of advice, and it can be hard to know which small habits deserve attention. "
        "A calm routine makes useful guidance easier to remember and easier to follow each day. "
        "Most people do not realize that small changes can matter more than dramatic changes made all at once. "
        "The good news is that a steady daily habit can make this easier and help you feel more confident. "
        "Keep the rest of the routine simple and give yourself time to notice what actually helps. "
        "Now pay attention because this next point matters when a symptom is new or suddenly becomes much worse. "
        "Do not ignore sudden chest pressure or severe shortness of breath, and seek urgent medical help when appropriate. "
        "After that serious point, return to the calm plan and write down any questions you want to discuss with your doctor. "
        "A simple written note can make the next conversation clearer and less stressful. "
        "Thankfully, many daily habits become easier once they are connected to the same time and place each day."
    )


def test_turbo_keeps_native_tags_and_original_uses_local_native_controls(tmp_path: Path, monkeypatch) -> None:
    # Avoid external FFmpeg mastering in this integration test; the dedicated audio
    # tests cover the mastering functions themselves.
    monkeypatch.setattr("queue_manager.master_professional_voice", lambda _path: {"target_lufs": -13.0})
    monkeypatch.setattr("queue_manager.create_video_master_48k_stereo", lambda src, dst: Path(dst).write_bytes(Path(src).read_bytes()))
    monkeypatch.setattr("queue_manager.write_caption_files", lambda _words, srt, vtt: (Path(srt).write_text("", encoding="utf-8"), Path(vtt).write_text("", encoding="utf-8")))

    async def run(model: str):
        model_root = tmp_path / model
        model_root.mkdir()
        engine = _CaptureEngine()
        manager = QueueManager(engine=engine, storage=_Storage(model_root))
        manager.quality = _AsrWarningQuality()
        request = AudioJobCreate(
            preset="Motivational Speech",
            audio_number=1,
            title=f"emotion {model}",
            text=_emotion_script(),
            voice_mode="default",
            options=GenerationOptions(model=model, split_text=True, chunk_words=85, platform_assets=False),
        )
        public = await manager.create(request, enqueue=False)
        job = manager.get_raw(public["id"])
        assert job["emotion_summary"]["applied_count"] >= 2
        await manager._process(job)
        assert job["status"] == "completed"
        assert job["quality_summary"]["retries"] == 0, job["quality_summary"]
        await manager.stop()
        return engine.calls

    turbo_calls = asyncio.run(run("chatterbox-turbo"))
    assert any("[happy]" in text or "[narration]" in text or "[dramatic]" in text or "[surprised]" in text for _, text, _ in turbo_calls)

    original_calls = asyncio.run(run("chatterbox"))
    assert all("[happy]" not in text and "[narration]" not in text and "[dramatic]" not in text and "[surprised]" not in text for _, text, _ in original_calls)
    emotion_options = [options for _, _, options in original_calls if options.get("_emotion_tag")]
    assert emotion_options
    assert any(options["exaggeration"] > 0.63 for options in emotion_options)


def test_pause_shaper_keeps_long_emotional_pause_longer_than_medium_pause() -> None:
    from professional_audio import shape_professional_pauses
    sr = 24000
    t = np.arange(int(sr * 0.30), dtype=np.float32) / sr
    tone = (0.08 * np.sin(2 * np.pi * 190 * t)).astype(np.float32)
    audio = np.concatenate([
        tone,
        np.zeros(int(sr * 0.80), dtype=np.float32),
        tone,
        np.zeros(int(sr * 1.60), dtype=np.float32),
        tone,
    ])
    shaped = shape_professional_pauses(audio, sr)
    silent = np.abs(shaped) < 1e-7
    runs = []
    start = None
    for i, value in enumerate(silent):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start > int(sr * 0.20):
                runs.append((i - start) / sr)
            start = None
    internal = sorted(runs)[:]
    assert len(internal) >= 2
    # The 1.6 s intentional pause is compacted, but remains perceptibly longer
    # than the ordinary 0.8 s pause. v1.5.3 accidentally inverted this relation.
    assert max(internal) > min(internal) + 0.08


def test_final_master_preserves_dynamic_contrast(tmp_path: Path) -> None:
    import soundfile as sf
    from professional_audio import master_professional_voice
    sr = 24000
    t = np.arange(sr * 3, dtype=np.float32) / sr
    neutral = (0.05 * np.sin(2 * np.pi * 180 * t) + 0.01 * np.sin(2 * np.pi * 360 * t)).astype(np.float32)
    expressive = (0.10 * np.sin(2 * np.pi * 180 * t) + 0.02 * np.sin(2 * np.pi * 360 * t)).astype(np.float32)
    path = tmp_path / "dynamic_voice.wav"
    sf.write(path, np.concatenate([neutral, expressive]), sr, subtype="PCM_16")
    master_professional_voice(path)
    mastered, out_sr = sf.read(path, dtype="float32")
    assert out_sr == sr
    before_db = _rms_db(mastered[: sr * 3])
    expressive_db = _rms_db(mastered[sr * 3 :])
    assert expressive_db - before_db > 2.0
    assert float(np.max(np.abs(mastered))) < 0.98
