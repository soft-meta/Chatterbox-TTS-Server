import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _profile() -> dict:
    tree = ast.parse((ROOT / "queue_manager.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "MOTIVATIONAL_TURBO_PROFILE":
                return ast.literal_eval(node.value)
    raise AssertionError("MOTIVATIONAL_TURBO_PROFILE not found")


def test_motivational_turbo_backend_profile_is_enforced() -> None:
    profile = _profile()
    assert profile["temperature"] == 0.60
    assert profile["top_p"] == 0.85
    assert profile["speed_factor"] == 1.0  # target-band pacing now owns the final speaking rate
    assert profile["chunk_words"] == 85
    assert profile["inter_chunk_pause_ms"] == 220
    assert profile["cfg_weight"] == 0.0
    assert profile["exaggeration"] == 0.0
    assert profile["min_p"] == 0.0


def test_profile_is_applied_only_to_turbo_motivational() -> None:
    source = (ROOT / "queue_manager.py").read_text(encoding="utf-8")
    assert 'request.preset == "Motivational Speech"' in source
    assert 'options.get("model") == "chatterbox-turbo"' in source
    assert "options.update(MOTIVATIONAL_TURBO_PROFILE)" in source


def test_browser_sends_preset_to_backend() -> None:
    source = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    assert "preset: tab.preset" in source


def test_pitch_preserving_final_tempo_is_used() -> None:
    source = (ROOT / "queue_manager.py").read_text(encoding="utf-8")
    assert '"-filter:a", f"atempo={speed_factor:.6f}"' in source
    engine = (ROOT / "engine.py").read_text(encoding="utf-8")
    assert "audio_functional.resample" not in engine
