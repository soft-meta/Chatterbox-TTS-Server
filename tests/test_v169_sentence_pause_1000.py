from pathlib import Path
import numpy as np
from models import GenerationOptions
from queue_manager import apply_sentence_end_pause

ROOT = Path(__file__).resolve().parents[1]

def test_model_accepts_1000_ms():
    assert GenerationOptions(sentence_end_pause_ms=1000).sentence_end_pause_ms == 1000

def test_ui_slider_exposes_1000_ms():
    html = (ROOT / 'ui' / 'index.html').read_text(encoding='utf-8')
    assert 'data-field="sentence_end_pause_ms"' in html
    assert 'data-field="sentence_end_pause_ms" type="range" min="0" max="1000" step="10"' in html

def test_pause_processor_honors_1000_ms():
    sr = 1000
    # 0.5 sec speech, 1.2 sec silence, 0.5 sec speech => one sentence boundary
    speech = np.ones(500, dtype=np.float32) * 0.1
    silence = np.zeros(1200, dtype=np.float32)
    audio = np.concatenate([speech, silence, speech])
    out = apply_sentence_end_pause(audio, sr, 'Sentence one. Sentence two.', 1000)
    # expected near 0.5 + 1.0 + 0.5 = 2.0 sec
    assert 1950 <= len(out) <= 2050
