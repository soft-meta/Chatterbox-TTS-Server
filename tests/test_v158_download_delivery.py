from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import sys
import types

stub = types.ModuleType("softmeta_chatterbox")
class _GenerationSettings:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
class _SoftMetaChatterboxEngine:
    def __init__(self, device="auto"):
        self.device = "cpu"
        self.model_name = None
    def status_dict(self): return {"device": self.device, "model_name": self.model_name, "loading": False, "error": None, "sample_rate": None}
    def load(self, _model): pass
    def unload(self): pass
stub.GenerationSettings = _GenerationSettings
stub.SoftMetaChatterboxEngine = _SoftMetaChatterboxEngine
sys.modules["softmeta_chatterbox"] = stub
if "engine" in sys.modules:
    sys.modules["engine"].SoftMetaChatterboxEngine = _SoftMetaChatterboxEngine
    sys.modules["engine"].GenerationSettings = _GenerationSettings

import server


class _Queue:
    def __init__(self, job: dict):
        self.job = job
    def get(self, job_id: str):
        if job_id != self.job['id']:
            raise KeyError(job_id)
        return dict(self.job)
    def get_raw(self, job_id: str):
        return self.get(job_id)


class _Storage:
    def __init__(self, root: Path):
        self.outputs = root
    def output_path(self, filename: str) -> Path:
        return self.outputs / Path(filename).name
    @staticmethod
    def media_type(path: Path) -> str:
        return 'audio/wav' if path.suffix.lower() == '.wav' else 'application/octet-stream'


def _write_wav(path: Path, seconds: float = 2.0, sr: int = 24000) -> None:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    sf.write(path, (0.05 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), sr, subtype='PCM_16')


def test_full_audio_download_uses_dedicated_attachment_route(tmp_path: Path, monkeypatch) -> None:
    filename = 'Audio_1_Final.wav'
    _write_wav(tmp_path / filename)
    job = {'id': 'job1', 'status': 'completed', 'output_filename': filename, 'title': 'Audio 1', 'audio_number': 1}
    monkeypatch.setattr(server, 'queue', _Queue(job))
    monkeypatch.setattr(server, 'storage', _Storage(tmp_path))

    response = server.job_audio_download('job1')
    disposition = response.headers.get('content-disposition', '')
    assert disposition.lower().startswith('attachment;')
    assert filename in disposition
    assert response.headers.get('cache-control') == 'no-store, no-cache, must-revalidate'
    assert response.headers.get('x-content-type-options') == 'nosniff'


def test_cut_response_returns_attachment_download_url_not_static_outputs(tmp_path: Path, monkeypatch) -> None:
    filename = 'Audio_1_Final.wav'
    _write_wav(tmp_path / filename, seconds=4.0)
    job = {'id': 'job1', 'status': 'completed', 'output_filename': filename, 'title': 'Audio 1', 'audio_number': 1}
    monkeypatch.setattr(server, 'queue', _Queue(job))
    monkeypatch.setattr(server, 'storage', _Storage(tmp_path))

    request = SimpleNamespace(start_seconds=0.0, end_seconds=2.0, filename_prefix='Part_One')
    data = asyncio.run(server.cut_audio('job1', request))
    assert data['filename'].endswith('.wav')
    assert data['download_url'].startswith('/api/outputs/')
    assert data['download_url'].endswith('/download')
    assert not data['download_url'].startswith('/outputs/')

    response = server.download_output(data['filename'])
    disposition = response.headers.get('content-disposition', '')
    assert disposition.lower().startswith('attachment;')
    assert data['filename'] in disposition


def test_ui_uses_server_attachment_routes_for_full_queue_and_cut_downloads() -> None:
    root = Path(__file__).resolve().parents[1]
    js = (root / 'ui' / 'app.js').read_text(encoding='utf-8')
    assert '/api/jobs/${job.id}/download' in js
    assert 'data.download_url' in js
    assert 'triggerAttachmentDownload' in js
    # The cut workflow must no longer download the static /outputs URL.
    assert 'anchor.href = data.url' not in js
