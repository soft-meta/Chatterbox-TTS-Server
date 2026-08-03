from pathlib import Path

import numpy as np

from models import VoiceDesignRequest
from voice_designer import VoiceDesigner
from voice_worker import evaluate_uniqueness

ROOT = Path(__file__).resolve().parents[1]


def test_v070_identity_profile_is_primary_and_stable() -> None:
    first = VoiceDesigner.build_profile(
        age=70, gender="male", language="en-US", emotion="warm",
        description="", seed=2025, candidate_index=0,
    )
    second = VoiceDesigner.build_profile(
        age=70, gender="male", language="en-US", emotion="warm",
        description="", seed=2025, candidate_index=1,
    )
    assert first.identity_code.startswith("SM-")
    assert first.identity_code != second.identity_code
    assert first.identity_traits["vocal_anatomy"] != second.identity_traits["vocal_anatomy"] or first.identity_traits["spectral_colour"] != second.identity_traits["spectral_colour"]
    assert "clearly distinct, stable human identity" in first.effective_description
    assert "Older age must not force a deep pitch" in first.effective_description
    assert "Do not make a young voice sound old by slowing" in first.effective_description


def test_v070_request_uses_stricter_default_threshold() -> None:
    request = VoiceDesignRequest(name="New Voice", sample_text="This is a test sentence.")
    assert request.uniqueness_threshold == 0.72


def test_v070_baseline_is_not_called_one_hundred_percent_different() -> None:
    result = evaluate_uniqueness(np.array([1.0, 0.0], dtype=np.float32), [], 0.72)
    assert result["status"] == "baseline"
    assert result["difference_score"] is None
    assert result["similarity_percent"] is None


def test_v070_similarity_rejects_repeated_identity() -> None:
    embedding = np.array([1.0, 0.0], dtype=np.float32)
    result = evaluate_uniqueness(embedding, [("Existing", embedding.copy())], 0.72)
    assert result["status"] == "too_similar"
    assert result["similarity_percent"] == 100.0


def test_v070_ui_reports_similarity_and_overgeneration() -> None:
    script = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    designer = (ROOT / "voice_designer.py").read_text(encoding="utf-8")
    worker = (ROOT / "voice_worker.py").read_text(encoding="utf-8")
    assert "similar · rejected" in script
    assert "Baseline candidate" in script
    assert "max_attempts = min(12" in designer
    assert "rejected_count" in worker
    assert "attempted_count" in worker
