import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import queue_manager
import server
from models import GenerationOptions


def tone(sr, ms, amp=0.18, hz=220.0):
    n = int(sr * ms / 1000)
    t = np.arange(n, dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def silence(sr, ms):
    return np.zeros(int(sr * ms / 1000), dtype=np.float32)


def test_pause_model_accepts_zero_and_600():
    assert GenerationOptions(sentence_end_pause_ms=0).sentence_end_pause_ms == 0
    assert GenerationOptions(sentence_end_pause_ms=600).sentence_end_pause_ms == 600


def test_sentence_pause_zero_vs_350_changes_duration_measurably():
    sr = 24000
    # Three spoken sentences with two clear internal punctuation pauses.
    audio = np.concatenate([
        tone(sr, 500), silence(sr, 420), tone(sr, 500),
        silence(sr, 360), tone(sr, 500),
    ])
    text = "One sentence. Two sentence. Three sentence."
    out0 = queue_manager.apply_sentence_end_pause(audio, sr, text, 0)
    out350 = queue_manager.apply_sentence_end_pause(audio, sr, text, 350)
    # Two sentence boundaries * 350ms means ~700ms difference.
    diff = (len(out350) - len(out0)) / sr
    assert 0.64 <= diff <= 0.76, diff


def test_sentence_pause_only_targets_expected_boundary_count():
    sr = 24000
    # Includes a shorter breath (120ms) plus two sentence pauses.
    audio = np.concatenate([
        tone(sr, 400), silence(sr, 120), tone(sr, 300), silence(sr, 430),
        tone(sr, 400), silence(sr, 390), tone(sr, 400),
    ])
    text = "Sentence one has a breath. Sentence two ends. Sentence three."
    out = queue_manager.apply_sentence_end_pause(audio, sr, text, 250)
    # Should preserve the 120ms breath while normalizing two longer sentence gaps.
    expected_ms = 400 + 120 + 300 + 250 + 400 + 250 + 400
    actual_ms = len(out) * 1000 / sr
    assert abs(actual_ms - expected_ms) <= 45, (actual_ms, expected_ms)


def test_initial_data_exposes_server_start_epoch():
    payload = server.initial_data()
    runtime = payload["runtime"]
    assert isinstance(runtime.get("server_started_at"), (int, float))
    assert runtime["server_started_at"] > 0


def test_disconnect_button_contains_runtime_timer_ui():
    html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    assert 'id="colab-runtime-timer"' in html
    assert "server_started_at" in js
    assert "updateColabRuntimeTimer" in js


def test_end_to_end_queue_pause_0_vs_350_changes_wav_duration(tmp_path, monkeypatch):
    import asyncio, types, torch, soundfile as sf
    from models import AudioJobCreate
    from queue_manager import QueueManager

    class Storage:
        def __init__(self, root):
            self.outputs = root / 'outputs'; self.outputs.mkdir(parents=True)
        def load_jobs(self): return {}
        def save_jobs(self, jobs): pass
        def output_path(self, filename): return self.outputs / filename
        def voice_path(self, kind, filename): return None

    class Engine:
        def generate(self, text, *, model_name, reference_audio, language, options):
            sr = 24000
            wav = np.concatenate([
                tone(sr, 450), silence(sr, 410), tone(sr, 450),
                silence(sr, 370), tone(sr, 450),
            ])
            return types.SimpleNamespace(waveform=torch.from_numpy(wav).unsqueeze(0), sample_rate=sr)

    async def render(pause_ms, title):
        mgr = QueueManager(Engine(), Storage(tmp_path / title))
        monkeypatch.setattr(mgr, '_normalise_loudness', lambda path: {'profile':'test'})
        req = AudioJobCreate(
            text='Sentence one. Sentence two. Sentence three.',
            title=title,
            voice_mode='default',
            options={'speed_factor':1.0, 'sentence_end_pause_ms': pause_ms},
        )
        pub = await mgr.create(req, enqueue=False)
        job = mgr.get_raw(pub['id'])
        await mgr._process(job)
        return sf.info(mgr.storage.output_path(job['output_filename'])).duration

    d0 = asyncio.run(render(0, 'zero'))
    d350 = asyncio.run(render(350, 'threefifty'))
    assert d350 - d0 > 0.64, (d0, d350)
