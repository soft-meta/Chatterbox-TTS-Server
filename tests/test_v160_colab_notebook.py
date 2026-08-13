import json
from pathlib import Path

NB=Path(__file__).resolve().parents[1] / 'colab' / 'SoftMeta_Chatterbox_TTS_Colab_v1.6.8.ipynb'


def test_colab_notebook_is_fresh_and_persistent():
    assert NB.exists(), 'v1.6.8 notebook has not been built yet'
    nb=json.loads(NB.read_text())
    joined='\n'.join(''.join(c.get('source',[])) for c in nb['cells'])
    assert 'v1.6.8' in joined
    assert "APP_VERSION == '1.6.8'" not in joined
    assert 'SOFTMETA_MIN_SERVER_VERSION="1.6.2"' in joined
    assert 'SOFTMETA_MAX_SERVER_VERSION_EXCLUSIVE="1.7.0"' in joined
    assert 'SOFTMETA_MODEL": "chatterbox-turbo"' in joined or 'SOFTMETA_MODEL"' in joined
    assert 'faster-whisper' not in joined.lower()
    assert 'speechbrain' not in joined.lower()
    assert 'SOFTMETA_ASR_MODEL' not in joined
    assert 'SOFTMETA_SPEAKER_CACHE' not in joined
    assert 'KEEP_RUNTIME_ACTIVE' in joined
    assert 'Server exited unexpectedly with code' in joined and 'restarting...' in joined
    # Notebook itself must never automatically unassign Colab after a job.
    assert 'runtime.unassign()' not in joined
    meta=nb.get('metadata',{})
    assert meta.get('accelerator')=='GPU'
    assert meta.get('colab',{}).get('gpuType')=='L4'
