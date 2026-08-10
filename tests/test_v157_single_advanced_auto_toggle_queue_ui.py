from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# queue_manager imports the separately installed adapter; keep unit tests self-contained.
stub = types.ModuleType('softmeta_chatterbox')
class _GenerationSettings:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
class _SoftMetaChatterboxEngine: pass
stub.GenerationSettings = _GenerationSettings
stub.SoftMetaChatterboxEngine = _SoftMetaChatterboxEngine
sys.modules.setdefault('softmeta_chatterbox', stub)

from emotion_director import analyze_turbo_avatar_performance, strip_turbo_tags
from models import AudioJobCreate, GenerationOptions
from queue_manager import QueueManager


class _Storage:
    def __init__(self, root: Path):
        self.root = root
        self.saved = {}
    def load_jobs(self): return dict(self.saved)
    def save_jobs(self, jobs): self.saved = {key: dict(value) for key, value in jobs.items()}
    def output_path(self, filename): return self.root / filename
    def voice_path(self, _mode, _filename): return None
    def delete_output_artifacts(self, _filename): pass
    def clear_outputs(self): pass


class _CaptureEngine:
    def __init__(self): self.calls = []
    def generate(self, text, *, model_name, reference_audio, language, options):
        self.calls.append((model_name, text, dict(options)))
        sr = 24000
        words = max(1, len(strip_turbo_tags(text).split()))
        seconds = max(0.9, words / 148.0 * 60.0)
        t = np.arange(int(sr * seconds), dtype=np.float32) / sr
        audio = (0.055 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        return SimpleNamespace(waveform=torch.from_numpy(audio).unsqueeze(0), sample_rate=sr)


def _request(text: str, *, auto_emotion: bool) -> AudioJobCreate:
    return AudioJobCreate(
        preset='Motivational Speech',
        audio_number=1,
        title='v157',
        text=text,
        voice_mode='default',
        auto_emotion=auto_emotion,
        options=GenerationOptions(
            model='chatterbox-turbo',
            split_text=True,
            chunk_words=85,
            quality_gate=False,
            speaker_consistency=False,
            platform_assets=False,
        ),
    )


def test_auto_emotion_is_opt_in_and_advanced_pipeline_is_the_single_generate_path(tmp_path: Path) -> None:
    text = (
        'My grandson made me laugh when he said I finally looked happier. '
        'I regret the years I spent worrying about things I could not control. '
        'Then I saw a result I never expected, and it stopped me in my tracks. '
    ) * 8
    manager = QueueManager(engine=SimpleNamespace(), storage=_Storage(tmp_path))

    async def run():
        off_public = await manager.create(_request(text, auto_emotion=False), enqueue=False)
        on_public = await manager.create(_request(text, auto_emotion=True), enqueue=False)
        off = manager.get_raw(off_public['id'])
        on = manager.get_raw(on_public['id'])
        await manager.stop()
        return off, on

    off, on = asyncio.run(run())
    assert off['generation_mode'] == 'advanced'
    assert off['auto_emotion'] is False
    assert off['emotion_summary'] is None
    assert not any(tag in off['generation_text'].lower() for tag in ('[chuckle]', '[laugh]', '[sigh]', '[gasp]'))
    assert on['generation_mode'] == 'advanced'
    assert on['auto_emotion'] is True
    assert on['emotion_summary'] is not None
    assert on['emotion_summary']['applied_count'] > 0
    assert any(tag in on['generation_text'].lower() for tag in ('[chuckle]', '[laugh]', '[sigh]', '[gasp]'))


def test_turbo_auto_planner_uses_only_four_events_and_medium_density() -> None:
    moments = [
        'My grandson made me laugh, and the whole room laughed with me.',
        'I smiled because his little joke was exactly what I needed that morning.',
        'I regret the years I spent carrying every problem alone.',
        'I wish I had learned that lesson much sooner.',
        'I never expected the test result, and it stopped me in my tracks.',
        'To my surprise, one small change made the next week feel completely different.',
    ]
    # Roughly seven minutes at senior-friendly narration speed with many genuine moments.
    text = ' '.join(moments * 15)
    result = analyze_turbo_avatar_performance(text)
    allowed = {'chuckle', 'laugh', 'sigh', 'gasp'}
    auto = [item for item in result.placements if item.get('source') != 'manual-event']
    assert auto
    assert {item['tag'] for item in auto} <= allowed
    # Medium use: slightly more than the old ~10 cues on a seven-minute script,
    # but nowhere near a tag every sentence.
    assert 11 <= len(auto) <= 14, len(auto)
    assert result.avatar_applied_count >= 9


def test_neutral_script_does_not_get_fake_laughter_or_sadness() -> None:
    text = (
        'Drink water with your meals. Keep your walking shoes near the door. '
        'Write your appointment time on the calendar. Sit down when you put on your socks. '
    ) * 30
    result = analyze_turbo_avatar_performance(text)
    assert result.applied_count == 0
    assert result.tagged_text == text.strip()


def test_completed_job_can_be_dismissed_from_monitor_without_deleting_audio_access(tmp_path: Path) -> None:
    storage = _Storage(tmp_path)
    manager = QueueManager(engine=SimpleNamespace(), storage=storage)
    async def run():
        public = await manager.create(_request('A short script for history.', auto_emotion=False), enqueue=False)
        job = manager.get_raw(public['id'])
        job['status'] = 'completed'
        job['output_filename'] = 'kept.wav'
        await manager.dismiss_from_monitor(job['id'])
        dismissed = manager.get_raw(job['id'])
        await manager.stop()
        return dismissed
    job = asyncio.run(run())
    assert job['monitor_dismissed'] is True
    assert job['output_filename'] == 'kept.wav'


def test_auto_emotion_off_vs_on_both_keep_professional_generation_but_only_on_sends_events(tmp_path: Path, monkeypatch) -> None:
    text = (
        'My grandson made me laugh when he said I finally looked happier. '
        'I regret the years I spent carrying every problem alone. '
        'Then I saw a result I never expected, and it stopped me in my tracks. '
        'I wrote the lesson down and shared it calmly with other people. '
    ) * 7
    mastered = []
    monkeypatch.setattr('queue_manager.apply_tempo_array', lambda audio, _sr, _factor: audio)
    monkeypatch.setattr('queue_manager.master_professional_voice', lambda path, **_kwargs: mastered.append(Path(path).name) or {'target_lufs': -13.0})
    monkeypatch.setattr('queue_manager.analyze_prosody_quality', lambda *_a, **_k: {'score': 90.0, 'rating': 'Strong'})

    async def render(enabled: bool, folder: str):
        root = tmp_path / folder
        root.mkdir()
        engine = _CaptureEngine()
        manager = QueueManager(engine=engine, storage=_Storage(root))
        public = await manager.create(_request(text, auto_emotion=enabled), enqueue=False)
        job = manager.get_raw(public['id'])
        await manager._process(job)
        await manager.stop()
        return job, engine.calls

    off_job, off_calls = asyncio.run(render(False, 'off'))
    on_job, on_calls = asyncio.run(render(True, 'on'))
    assert off_job['status'] == on_job['status'] == 'completed'
    assert len(mastered) == 2  # professional mastering runs in both modes
    assert off_job['emotion_summary'] is None
    assert not any(any(tag in text.lower() for tag in ('[chuckle]', '[laugh]', '[sigh]', '[gasp]')) for _m, text, _o in off_calls)
    assert on_job['emotion_summary']['applied_count'] > 0
    sent_on = ' '.join(text for _m, text, _o in on_calls).lower()
    assert any(tag in sent_on for tag in ('[chuckle]', '[laugh]', '[sigh]', '[gasp]'))


def test_ui_has_one_generate_button_default_off_auto_emotion_and_draggable_progress() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / 'ui' / 'index.html').read_text(encoding='utf-8')
    js = (root / 'ui' / 'app.js').read_text(encoding='utf-8')
    css = (root / 'ui' / 'styles.css').read_text(encoding='utf-8')

    assert 'data-action="generate-audio"' in html
    assert 'Generate Advanced Audio' not in html
    assert 'data-action="generate-standard"' not in html
    assert 'data-action="generate-advanced"' not in html
    assert 'Turbo A/B Comparison' not in html

    assert 'data-action="toggle-auto-emotion"' in html
    assert 'aria-pressed="false"' in html
    assert 'auto_emotion: false' in js
    assert "auto_emotion: Boolean(tab.auto_emotion)" in js

    for tag in ('[chuckle]', '[laugh]', '[sigh]', '[gasp]'):
        assert f'data-event-tag="{tag}"' in html
    assert 'data-event-tag="[clear throat]"' not in html

    assert 'dismissQueueHistory' in js
    assert '/dismiss-monitor' in js
    assert 'queue-remove-history' in css

    assert 'floating-progress-track' in js
    assert 'floating-progress-fill' in js
    assert 'wireFloatingProgressDrag' in js
    assert 'pointerdown' in js and 'pointermove' in js and 'pointerup' in js
    assert 'FLOATING_POSITION_KEY' in js
