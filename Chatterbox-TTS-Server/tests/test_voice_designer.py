from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf

from voice_designer import VoiceDesigner


def _fake_worker(path: Path) -> None:
    path.write_text(
        """
import argparse, json
from pathlib import Path
import numpy as np
import soundfile as sf
parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
args = parser.parse_args()
data = json.loads(Path(args.request).read_text())
out = Path(data['output_path'])
sf.write(out, np.zeros(2400, dtype=np.float32), 24000, subtype='PCM_16')
print(json.dumps({'ok': True}))
""",
        encoding="utf-8",
    )


def test_voice_designer_uses_isolated_runner(tmp_path: Path) -> None:
    worker = tmp_path / "fake_worker.py"
    _fake_worker(worker)
    designer = VoiceDesigner(
        tmp_path / "voices",
        device="cpu",
        python_executable=sys.executable,
        worker_path=worker,
        timeout_seconds=30,
    )
    result = designer.generate(
        name="Test Voice",
        age=70,
        gender="male",
        language="en-US",
        emotion="reflective",
        description="Warm and intimate.",
        sample_text="This is a short test.",
        seed=2025,
    )
    assert result.path.exists()
    assert result.path.name.startswith("Test_Voice_2025")
    assert sf.info(result.path).samplerate == 24000
    assert result.profile.age == 70
    assert result.profile.gender == "male"
    assert result.profile.recommended_speed_factor == 0.84
    assert result.path.with_suffix(".json").exists()


def test_age_profiles_become_progressively_slower() -> None:
    speeds = [
        VoiceDesigner.build_profile(
            age=age,
            gender="female",
            language="en-US",
            emotion="warm",
            description="",
            seed=1,
        ).recommended_speed_factor
        for age in (50, 60, 70, 80, 90)
    ]
    assert speeds == sorted(speeds, reverse=True)
    assert speeds[0] == 0.96
    assert speeds[-1] == 0.68


def test_natural_voice_formula_enforces_us_english_and_human_style() -> None:
    profile = VoiceDesigner.build_profile(
        age=85,
        gender="female",
        language="en-US",
        emotion="calm",
        description="Soft and sincere.",
        seed=2025,
    )
    prompt = profile.effective_description
    assert "85-year-old American woman" in prompt
    assert "General American English" in prompt
    assert "AI assistant" in prompt
    assert "very slow" in prompt
    assert "Soft and sincere" in prompt


def test_voice_designer_reports_missing_environment(tmp_path: Path) -> None:
    designer = VoiceDesigner(
        tmp_path / "voices",
        python_executable=str(tmp_path / "missing-python"),
        worker_path=tmp_path / "missing-worker.py",
    )
    try:
        designer.generate(
            name="Test",
            description="A natural adult speaker with a clear voice.",
            sample_text="Testing.",
            seed=1,
        )
    except RuntimeError as error:
        assert "isolated Generate Voice environment is missing" in str(error)
    else:
        raise AssertionError("Expected a missing environment error")
