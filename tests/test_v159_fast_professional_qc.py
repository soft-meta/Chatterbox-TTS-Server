from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

stub = types.ModuleType('softmeta_chatterbox')
class _GenerationSettings:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
class _SoftMetaChatterboxEngine: pass
stub.GenerationSettings = _GenerationSettings
stub.SoftMetaChatterboxEngine = _SoftMetaChatterboxEngine
sys.modules.setdefault('softmeta_chatterbox', stub)

from emotion_director import strip_turbo_tags
from models import AudioJobCreate, GenerationOptions
from queue_manager import QueueManager
from quality_control import ChunkQualityReport
from speech_pipeline import build_long_form_segments
from emotion_director import is_heading


class _Storage:
    def __init__(self, root: Path): self.root = root
    def load_jobs(self): return {}
    def save_jobs(self, jobs): pass
    def output_path(self, filename): return self.root / filename
    def voice_path(self, _mode, _filename): return None


class _Engine:
    def __init__(self): self.calls = []
    def generate(self, text, *, model_name, reference_audio, language, options):
        self.calls.append(text)
        sr = 24000
        words = max(1, len(strip_turbo_tags(text).split()))
        seconds = words / 150.0 * 60.0
        t = np.arange(max(1, int(sr * seconds)), dtype=np.float32) / sr
        audio = (0.055 * np.sin(2*np.pi*180*t)).astype(np.float32)
        return SimpleNamespace(waveform=torch.from_numpy(audio).unsqueeze(0), sample_rate=sr)


def _request(text: str) -> AudioJobCreate:
    return AudioJobCreate(
        preset='Motivational Speech', audio_number=1, title='fast', text=text,
        voice_mode='default', auto_emotion=False,
        options=GenerationOptions(model='chatterbox-turbo', split_text=True, chunk_words=85,
                                  quality_gate=True, speaker_consistency=True, platform_assets=False),
    )


def test_turbo_advanced_uses_fast_qc_profile_by_default(tmp_path: Path):
    manager = QueueManager(engine=SimpleNamespace(), storage=_Storage(tmp_path))
    request = _request('A calm useful sentence for older listeners. ' * 25)
    options = manager._effective_options(request)
    assert options.get('_fast_professional_qc') is True
    asyncio.run(manager.stop())


def test_turbo_native_event_packing_does_not_force_one_model_call_per_tag_sentence():
    text = (
        'I was sitting with my grandson when he made a joke. [chuckle] I laughed and told him he was right. '
        'Then I explained the lesson slowly because it matters after seventy. '
        'I remembered the years I worried about things I could not change. [sigh] That was a hard lesson for me. '
        'But I also learned how one small habit can make the next day easier. '
    ) * 5
    segments = build_long_form_segments(text, max_words=85, heading_detector=is_heading,
                                       turbo_avatar_mode=True, avatar_window_words=700)
    total_words = len(strip_turbo_tags(text).split())
    # Efficient Turbo packing should be close to normal long-form chunk count,
    # not explode into one tiny generation for every event sentence.
    assert len(segments) <= max(4, int(total_words / 50) + 2), (total_words, len(segments))
    merged = ' '.join(seg.text for seg in segments)
    assert '[chuckle]' in merged and '[sigh]' in merged


def test_soft_asr_advisory_does_not_trigger_chunk_regeneration_in_fast_turbo(tmp_path: Path, monkeypatch):
    text = ('A steady routine can make the day easier. Keep the advice simple and practical. ' * 28)
    engine = _Engine()
    manager = QueueManager(engine=engine, storage=_Storage(tmp_path))

    # v1.5.9 fast path should not call the heavyweight per-chunk evaluator at all.
    def forbidden(*_a, **_k):
        raise AssertionError('heavy per-chunk ASR/speaker evaluate() should not run')
    manager.quality.evaluate = forbidden  # type: ignore[method-assign]
    monkeypatch.setattr('queue_manager.master_professional_voice', lambda path, **kwargs: {'target_lufs': -13.0})
    monkeypatch.setattr('queue_manager.analyze_prosody_quality', lambda *_a, **_k: {'score': 90.0, 'rating': 'Strong'})

    # One final verification is allowed and provides caption/QC metadata.
    final_calls = []
    def final_verify(path, expected_text, reference_path=None, speaker_check=True):
        final_calls.append((Path(path), expected_text, reference_path, speaker_check))
        return {
            'report': None,
            'words': [],
            'summary': {
                'checked_chunks': 0, 'passed_chunks': 0, 'warning_chunks': 0,
                'hard_failures': 0, 'retries': 0, 'average_score': 96.0,
                'average_wer': 0.08, 'minimum_speaker_similarity': 0.84,
                'asr_verified': True, 'speaker_verified': True, 'warnings': [],
                'mode': 'fast-final-pass',
            },
        }
    manager.quality.evaluate_final_file = final_verify  # type: ignore[attr-defined]

    async def run():
        public = await manager.create(_request(text), enqueue=False)
        job = manager.get_raw(public['id'])
        await manager._process(job)
        await manager.stop()
        return job
    job = asyncio.run(run())
    expected_segments = build_long_form_segments(job['generation_text'], max_words=85, heading_detector=is_heading,
                                                turbo_avatar_mode=True, avatar_window_words=0)
    assert job['status'] == 'completed'
    assert len(engine.calls) == len(expected_segments)
    assert len(final_calls) == 1
    assert job['quality_summary']['mode'] == 'fast-final-pass'
    assert job['quality_summary']['retries'] == 0


def test_fast_turbo_allows_only_one_retry_for_objective_bad_chunk(tmp_path: Path, monkeypatch):
    text = ('A calm sentence for a safe generation check. ' * 24)
    engine = _Engine()
    manager = QueueManager(engine=engine, storage=_Storage(tmp_path))
    monkeypatch.setattr('queue_manager.master_professional_voice', lambda path, **kwargs: {'target_lufs': -13.0})
    monkeypatch.setattr('queue_manager.analyze_prosody_quality', lambda *_a, **_k: {'score': 90.0})
    manager.quality.evaluate_final_file = lambda *_a, **_k: {'report': None, 'words': [], 'summary': {'checked_chunks': 1, 'passed_chunks': 1, 'warning_chunks': 0, 'hard_failures': 0, 'retries': 1, 'average_score': 95.0, 'average_wer': None, 'minimum_speaker_similarity': None, 'asr_verified': False, 'speaker_verified': False, 'warnings': [], 'mode': 'fast-final-pass'}}  # type: ignore[attr-defined]

    calls = {'n': 0}
    def fast_check(audio, sr, spoken):
        calls['n'] += 1
        # Fail the first generated chunk once, then accept everything.
        hard = calls['n'] == 1
        return ChunkQualityReport(passed=not hard, hard_failure=hard, retry_recommended=hard,
                                  reasons=['mostly silent audio'] if hard else [], score=100.0, metrics={})
    manager.quality.evaluate_acoustic = fast_check  # type: ignore[attr-defined]

    async def run():
        public = await manager.create(_request(text), enqueue=False)
        job = manager.get_raw(public['id'])
        await manager._process(job)
        await manager.stop()
        return job
    job = asyncio.run(run())
    base_segments = build_long_form_segments(job['generation_text'], max_words=85, heading_detector=is_heading,
                                            turbo_avatar_mode=True, avatar_window_words=0)
    assert job['status'] == 'completed'
    assert len(engine.calls) == len(base_segments) + 1


def test_final_asr_uses_one_batched_pass_with_fast_decode(tmp_path: Path):
    from quality_control import QualityController

    class _Word:
        start = 0.1
        end = 0.4
        word = ' hello'
    class _Segment:
        text = 'hello world'
        words = [_Word()]
    class _Batched:
        def __init__(self): self.calls = []
        def transcribe(self, path, **kwargs):
            self.calls.append((path, dict(kwargs)))
            return iter([_Segment()]), SimpleNamespace()

    wav = tmp_path / 'final.wav'
    import soundfile as sf
    sf.write(wav, np.zeros(24000, dtype=np.float32), 24000)
    qc = QualityController()
    qc._asr = object()
    batched = _Batched()
    qc._batched_asr = batched
    transcript, words, available = qc.transcribe_file_fast(wav, 'hello world')
    assert available is True
    assert transcript == 'hello world'
    assert words and words[0]['word'].strip() == 'hello'
    assert len(batched.calls) == 1
    kwargs = batched.calls[0][1]
    assert kwargs['batch_size'] == 8
    assert kwargs['beam_size'] == 1
    assert kwargs['word_timestamps'] is True
    assert kwargs['condition_on_previous_text'] is False


def test_acoustic_gate_never_invokes_asr_or_speaker_models():
    from quality_control import QualityController
    qc = QualityController()
    qc.transcribe = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('ASR called'))  # type: ignore[method-assign]
    qc.speaker_similarity = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('speaker called'))  # type: ignore[method-assign]
    sr = 24000
    t = np.arange(sr * 3, dtype=np.float32) / sr
    audio = (0.06 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    report = qc.evaluate_acoustic(audio, sr, 'A calm sentence with several words for this test.')
    assert report.passed is True
    assert report.asr_available is False
    assert report.speaker_check_available is False
