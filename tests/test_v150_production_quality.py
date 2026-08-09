from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

# QueueManager normally imports the separately installed engine adapter.
stub = types.ModuleType("softmeta_chatterbox")
class _GenerationSettings:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
class _SoftMetaChatterboxEngine: pass
stub.GenerationSettings = _GenerationSettings
stub.SoftMetaChatterboxEngine = _SoftMetaChatterboxEngine
sys.modules.setdefault("softmeta_chatterbox", stub)

from captions import write_caption_files
from models import AudioJobCreate, GenerationOptions
from professional_audio import adaptive_tempo_factor
from pronunciation_engine import prepare_pronunciation_text
from quality_control import ChunkQualityReport, QualityController, word_error_rate
from queue_manager import QueueManager
from reference_quality import analyze_reference_voice
from speech_pipeline import build_long_form_segments, pace_profile
from emotion_director import is_heading

ROOT = Path(__file__).resolve().parents[1]


def test_pronunciation_engine_handles_high_risk_senior_terms_and_preserves_tags() -> None:
    text = "[narration] At age 70+, an A1C of 6.5% and blood pressure of 120/80 may matter. Take 5 mg of B12 only as advised."
    spoken = prepare_pronunciation_text(text)
    assert "[narration]" in spoken
    assert "seventy and older" in spoken
    assert "A one C" in spoken
    assert "six point five percent" in spoken
    assert "one hundred twenty over eighty" in spoken
    assert "five milligrams" in spoken
    assert "B twelve" in spoken


def test_age_profiles_and_target_band_avoid_unnecessary_tempo_processing() -> None:
    assert pace_profile("60s")["body"] > pace_profile("70s")["body"] > pace_profile("80s")["body"] > pace_profile("90plus")["body"]
    segments = build_long_form_segments(
        "Number One: A Calm Habit\nA steady routine can help older adults understand advice without rushing through the explanation.",
        max_words=85,
        heading_detector=is_heading,
        age_profile="80s",
    )
    assert segments[0].min_wpm < segments[0].target_wpm < segments[0].max_wpm
    assert adaptive_tempo_factor(current_wpm=segments[0].target_wpm + 2, target_wpm=segments[0].target_wpm,
                                 min_wpm=segments[0].min_wpm, max_wpm=segments[0].max_wpm) == 1.0


def test_reference_quality_distinguishes_clean_and_bad_samples(tmp_path: Path) -> None:
    sr = 16000
    # Ten seconds with clear speech-like energy and modest room floor.
    t = np.arange(sr * 10, dtype=np.float32) / sr
    good = np.full_like(t, 0.002)
    speech = (np.sin(2 * np.pi * 180 * t) * 0.09).astype(np.float32)
    gate = ((t % 1.0) < 0.72).astype(np.float32)
    good += speech * gate
    good_path = tmp_path / "good.wav"
    sf.write(good_path, good, sr)
    good_report = analyze_reference_voice(good_path)
    assert good_report["usable"] is True
    assert good_report["score"] >= 70

    bad_path = tmp_path / "bad.wav"
    sf.write(bad_path, np.full(sr * 3, 0.0002, dtype=np.float32), sr)
    bad_report = analyze_reference_voice(bad_path)
    assert bad_report["score"] < good_report["score"]
    assert bad_report["rating"] in {"Poor", "Fair"}


def test_quality_gate_detects_transcript_error_and_accepts_clean_match() -> None:
    sr = 24000
    expected = "This calm sentence contains ten simple words for the listener today."
    seconds = len(expected.split()) / 150.0 * 60.0
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    audio = (0.08 * np.sin(2 * np.pi * 190 * t)).astype(np.float32)
    qc = QualityController()
    qc.transcribe = lambda *_args, **_kwargs: (expected, [], True)  # type: ignore[method-assign]
    qc.speaker_similarity = lambda *_args, **_kwargs: (0.55, True)  # type: ignore[method-assign]
    passed = qc.evaluate(audio, sr, expected, Path("reference.wav"), [0.54, 0.56])
    assert passed.passed is True
    assert passed.wer == 0.0

    # A moderate verifier disagreement now requests a retry without hard-failing
    # the whole long-form job.
    qc.transcribe = lambda *_args, **_kwargs: ("This calm phrase has ten clear words for older listener today", [], True)  # type: ignore[method-assign]
    uncertain = qc.evaluate(audio, sr, expected, Path("reference.wav"), [0.54, 0.56])
    assert uncertain.passed is True
    assert uncertain.retry_recommended is True
    assert word_error_rate(expected, uncertain.transcript) > 0.3

    # Truly severe content loss still fails when ASR evidence is overwhelming.
    qc.transcribe = lambda *_args, **_kwargs: ("This sentence", [], True)  # type: ignore[method-assign]
    failed = qc.evaluate(audio[: max(1, len(audio) // 3)], sr, expected, Path("reference.wav"), [0.54, 0.56])
    assert failed.passed is False
    assert failed.hard_failure is True


def test_original_uses_same_visible_direction_without_speaking_turbo_tags(tmp_path: Path) -> None:
    class _Storage:
        def load_jobs(self): return {}
        def save_jobs(self, jobs): pass
    manager = QueueManager(engine=SimpleNamespace(), storage=_Storage())
    try:
        spoken, options = manager._original_chunk_direction(
            "[happy] The good news is that this can improve with a steady routine.",
            {"temperature": 0.72, "exaggeration": 0.58, "cfg_weight": 0.35},
        )
        assert "[happy]" not in spoken
        assert spoken.startswith("The good news")
        assert options["exaggeration"] > 0.58
        assert options["cfg_weight"] <= 0.32
        assert options["_emotion_tag"] == "happy"
        retry_spoken, retry_options = manager._original_chunk_direction(
            "[dramatic] This serious warning deserves calm attention.",
            {"temperature": 0.72, "exaggeration": 0.58, "cfg_weight": 0.35, "_quality_retry": True},
        )
        assert "[dramatic]" not in retry_spoken
        assert retry_options["temperature"] < 0.60
        assert retry_options["exaggeration"] > 0.52
        assert retry_options["cfg_weight"] <= 0.28
        assert retry_options["_emotion_tag"] == "dramatic"
    finally:
        manager.executor.shutdown(wait=False, cancel_futures=True)


def test_caption_exports_are_valid_srt_and_vtt(tmp_path: Path) -> None:
    words = [
        {"start": 0.0, "end": 0.4, "word": "This"},
        {"start": 0.4, "end": 0.8, "word": " is"},
        {"start": 0.8, "end": 1.2, "word": " clear"},
        {"start": 1.2, "end": 1.6, "word": " advice."},
    ]
    srt = tmp_path / "captions.srt"
    vtt = tmp_path / "captions.vtt"
    write_caption_files(words, srt, vtt)
    assert "00:00:00,000 -->" in srt.read_text(encoding="utf-8")
    assert vtt.read_text(encoding="utf-8").startswith("WEBVTT\n")
    assert "This is clear advice." in srt.read_text(encoding="utf-8")


def test_ui_and_dependencies_expose_production_quality_for_original_and_turbo() -> None:
    html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    app = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    server = (ROOT / "server.py").read_text(encoding="utf-8")
    req = (ROOT / "requirements-colab.txt").read_text(encoding="utf-8")
    assert "Senior Listener Pace" in html
    assert "Video Master 48k" in html and "SRT" in html and "VTT" in html
    assert 'data-role="production-quality"' in html
    assert "['chatterbox-turbo', 'chatterbox'].includes(model)" in app
    assert '"models": ["chatterbox", "chatterbox-turbo"]' in server
    assert "faster-whisper==1.2.1" in req
    assert "speechbrain==1.1.0" in req


class _Storage:
    def __init__(self, root: Path):
        self.root = root
        self.saved = {}
    def load_jobs(self): return {}
    def save_jobs(self, jobs): self.saved = dict(jobs)
    def output_path(self, filename): return self.root / filename
    def voice_path(self, mode, filename): return None


class _CaptureEngine:
    def __init__(self): self.calls = []
    def generate(self, text, *, model_name, reference_audio, language, options):
        self.calls.append((model_name, text, dict(options)))
        words = max(1, len(text.split()))
        sr = 24000
        seconds = words / 150.0 * 60.0
        t = np.arange(max(1, int(sr * seconds)), dtype=np.float32) / sr
        audio = (0.065 * np.sin(2 * np.pi * 185 * t) + 0.008 * np.sin(2 * np.pi * 2400 * t)).astype(np.float32)
        return SimpleNamespace(waveform=torch.from_numpy(audio).unsqueeze(0), sample_rate=sr)


class _PassingQuality:
    def evaluate(self, audio, sample_rate, expected_text, reference_path, speaker_history):
        return ChunkQualityReport(
            passed=True, score=98.0, transcript=expected_text, wer=0.0,
            speaker_similarity=None, metrics={"wpm": 150.0}, words=[],
            asr_available=True, speaker_check_available=False,
        )


def test_mock_end_to_end_assets_and_qc_are_shared_by_original_and_turbo(tmp_path: Path) -> None:
    async def run_model(model: str):
        root = tmp_path / model
        root.mkdir()
        engine = _CaptureEngine()
        manager = QueueManager(engine=engine, storage=_Storage(root))
        manager.quality = _PassingQuality()  # avoid model downloads in unit tests
        text = (
            "Why This Habit Matters After 70\n\n"
            "Many older adults overlook this simple habit. The good news is that a calm daily routine can make the advice easier to remember. "
            "Remember this important point because small steady changes are often easier to keep than sudden changes."
        )
        request = AudioJobCreate(
            preset="Motivational Speech", audio_number=1, title=f"{model} QC", text=text,
            voice_mode="default",
            options=GenerationOptions(model=model, split_text=True, chunk_words=85, senior_pace_profile="70s"),
        )
        public = await manager.create(request, enqueue=False)
        job = manager.get_raw(public["id"])
        await manager._process(job)
        assert job["status"] == "completed"
        assert job["quality_summary"]["checked_chunks"] >= 1
        assert job["quality_summary"]["average_score"] == 98.0
        video = root / job["video_master_filename"]
        assert video.exists()
        vinfo = sf.info(video)
        assert vinfo.samplerate == 48000 and vinfo.channels == 2
        assert (root / job["srt_filename"]).exists()
        assert (root / job["vtt_filename"]).exists()
        if model == "chatterbox":
            assert all("[happy]" not in call[1] and "[narration]" not in call[1] and "[surprised]" not in call[1] and "[dramatic]" not in call[1] for call in engine.calls)
        await manager.stop()

    asyncio.run(run_model("chatterbox-turbo"))
    asyncio.run(run_model("chatterbox"))

class _FailingQuality:
    def evaluate(self, audio, sample_rate, expected_text, reference_path, speaker_history):
        return ChunkQualityReport(
            passed=False, score=32.0, reasons=["high ASR mismatch (60%)"],
            transcript="wrong words", wer=0.6, speaker_similarity=None,
            metrics={"wpm": 150.0}, words=[], asr_available=True,
            speaker_check_available=False,
        )


def test_production_gate_does_not_ship_known_bad_chunk_after_retries(tmp_path: Path) -> None:
    async def run():
        engine = _CaptureEngine()
        manager = QueueManager(engine=engine, storage=_Storage(tmp_path))
        manager.quality = _FailingQuality()
        request = AudioJobCreate(
            preset="Motivational Speech", audio_number=1, title="Reject Bad Chunk",
            text="The good news is that this serious advice can be explained clearly and calmly for an older listener today.",
            voice_mode="default", options=GenerationOptions(model="chatterbox", split_text=True),
        )
        public = await manager.create(request, enqueue=False)
        job = manager.get_raw(public["id"])
        try:
            await manager._process(job)
        except RuntimeError as exc:
            assert "Production Quality Gate rejected chunk" in str(exc)
        else:
            raise AssertionError("Known-bad chunk was allowed through the production gate")
        assert len(engine.calls) >= 3  # original passes plus verified rescue attempts
        assert not list(tmp_path.glob("Audio_*.wav"))
        assert job["quality_summary"]["retries"] >= 2
        await manager.stop()
    asyncio.run(run())


def test_original_preserves_user_cfg_and_exaggeration_while_sharing_quality_pipeline() -> None:
    request = AudioJobCreate(
        preset="Motivational Speech", audio_number=1, title="Original controls",
        text="This is a calm test sentence for an older listener.", voice_mode="default",
        options=GenerationOptions(
            model="chatterbox", exaggeration=0.77, cfg_weight=0.28,
            temperature=0.83, speed_factor=0.97,
        ),
    )
    options = QueueManager._effective_options(request)
    assert options["exaggeration"] == 0.77
    assert options["cfg_weight"] == 0.28
    assert options["temperature"] == 0.83
    assert options["speed_factor"] == 0.97
    assert options["chunk_words"] == 85


def test_v150_colab_keeps_l4_and_verifies_qc_packages() -> None:
    path = ROOT / "colab" / "SoftMeta_Chatterbox_TTS_Colab_v1.5.4.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Faster-Whisper" in source
    assert 'version("faster-whisper")' in source
    assert 'version("speechbrain")' in source
    assert '"SOFTMETA_ASR_MODEL": "small.en"' in source
    assert '"SOFTMETA_SPEAKER_CACHE": "/content/hf_home/softmeta-speaker"' in source
