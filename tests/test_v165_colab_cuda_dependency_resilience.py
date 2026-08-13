from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "colab" / "SoftMeta_Chatterbox_TTS_Colab_v1.6.7.ipynb"

def text():
    nb=json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c.get("source", [])) for c in nb["cells"])

def test_torch_uses_dual_index_and_cudnn_fallback():
    t=text()
    assert "--extra-index-url https://pypi.org/simple" in t
    assert "nvidia-cudnn-cu12==9.1.0.70" in t
    assert "install_torch_cuda" in t

def test_chatterbox_does_not_reresolve_torch_cuda():
    t=text()
    assert '--no-deps "chatterbox-tts==0.1.7"' in t
    assert '"transformers==5.2.0"' in t
    assert '"librosa==0.11.0"' in t
    assert '"resemble-perth>=1.0.0"' in t

def test_notebook_version_is_v165():
    t=text()
    assert "v1.6.7" in t
    assert "if sys.argv[1] != '1.6.7'" in t
