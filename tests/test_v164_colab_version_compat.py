import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / 'colab' / 'SoftMeta_Chatterbox_TTS_Colab_v1.6.7.ipynb'


def joined_text():
    data = json.loads(NB.read_text(encoding='utf-8'))
    return '\n'.join(''.join(c.get('source', [])) for c in data['cells'])


def test_colab_does_not_abort_on_exact_server_version_mismatch():
    text = joined_text()
    assert "assert server.APP_VERSION == '1.6.7'" not in text
    assert 'SOFTMETA_MIN_SERVER_VERSION' in text
    assert 'Compatible GitHub server version' in text
    assert 'Notebook/server version difference is compatible' in text


def test_server_compatibility_is_checked_before_heavy_torch_install():
    text = joined_text()
    compat = text.index('SOFTMETA_MIN_SERVER_VERSION')
    heavy = text.index('torch==2.6.0 torchaudio==2.6.0')
    assert compat < heavy


def test_install_verifies_turbo_class_and_supported_sampling_signature():
    text = joined_text()
    assert 'from chatterbox.tts_turbo import ChatterboxTurboTTS' in text
    assert 'inspect.signature(ChatterboxTurboTTS.generate)' in text
    for name in ('temperature', 'top_p', 'top_k', 'repetition_penalty'):
        assert name in text
