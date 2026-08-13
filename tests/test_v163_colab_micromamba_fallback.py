import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / 'colab' / 'SoftMeta_Chatterbox_TTS_Colab_v1.6.8.ipynb'


def notebook_text():
    data = json.loads(NB.read_text(encoding='utf-8'))
    return '\n'.join(''.join(cell.get('source', [])) for cell in data['cells'])


def test_colab_has_official_micromamba_primary_and_github_fallback():
    text = notebook_text()
    assert 'https://micro.mamba.pm/api/micromamba/linux-64/latest' in text
    assert 'https://github.com/mamba-org/micromamba-releases/releases/download/' in text
    assert 'Micromamba primary download failed; trying GitHub fallback' in text


def test_colab_validates_micromamba_before_reuse():
    text = notebook_text()
    assert 'if [[ -x "$MICROMAMBA" ]] && ! "$MICROMAMBA" --version' in text
    assert 'rm -f "$MICROMAMBA"' in text
    assert 'Unable to install Micromamba from both official sources' in text
