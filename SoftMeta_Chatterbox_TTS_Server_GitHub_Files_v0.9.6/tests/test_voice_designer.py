from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from voice_designer import VoiceDesigner


def _fake_worker(path: Path) -> None:
    path.write_text(
        r'''
import argparse, json
from pathlib import Path
import numpy as np
import soundfile as sf
parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
args = parser.parse_args()
data = json.loads(Path(args.request).read_text())
session = Path(data['session_dir'])
session.mkdir(parents=True, exist_ok=True)
candidates = []
for index, profile in enumerate(data['profiles'][:data['candidate_count']], start=1):
    output = session / f'candidate_{index}.wav'
    sf.write(output, np.zeros(2400, dtype=np.float32), 24000, subtype='PCM_16')
    metadata = {
        **profile,
        'filename': output.name,
        'candidate_number': index,
        'seed': data['base_seed'] + (index - 1) * 104729,
        'duration': 0.1,
        'size': output.stat().st_size,
        'uniqueness': {
            'checked': True,
            'difference_score': 88.0,
            'max_similarity': 0.12,
            'closest_voice': None,
            'status': 'unique',
        },
    }
    output.with_suffix('.json').write_text(json.dumps(metadata))
    candidates.append(metadata)
print('SOFTMETA_RESULT=' + json.dumps({
    'ok': True,
    'session_id': data['session_id'],
    'model_id': data['model_id'],
    'candidate_count': len(candidates),
    'candidates': candidates,
    'uniqueness_threshold': data['uniqueness_threshold'],
}))
''',
        encoding="utf-8",
    )


def test_voice_designer_generates_and_saves_candidates(tmp_path: Path) -> None:
    worker = tmp_path / "fake_worker.py"
    _fake_worker(worker)
    designer = VoiceDesigner(
        tmp_path / "voices",
        tmp_path / "candidates",
        device="cpu",
        python_executable=sys.executable,
        worker_path=worker,
        timeout_seconds=30,
    )
    result = designer.generate_candidates(
        name="Test Voice",
        age=70,
        gender="male",
        language="en-US",
        emotion="reflective",
        description="Warm and intimate.",
        sample_text="This is a short test.",
        seed=2025,
        candidate_count=3,
    )
    assert result["candidate_count"] == 3
    assert result["candidates"][0]["age"] == 70
    session = result["session_id"]
    saved = designer.save_candidate(
        session_id=session,
        filename=result["candidates"][0]["filename"],
        voice_name="Chosen Voice",
    )
    assert saved.exists()
    assert saved.name == "Chosen_Voice.wav"
    assert sf.info(saved).samplerate == 24000
    assert saved.with_suffix(".json").exists()


def test_age_profiles_use_natural_cadence_without_global_slowdown() -> None:
    profiles = [
        VoiceDesigner.build_profile(
            age=age,
            gender="female",
            language="en-US",
            emotion="warm",
            description="",
            seed=1,
            candidate_index=0,
        )
        for age in (50, 60, 70, 80, 90)
    ]
    assert all(profile.recommended_speed_factor == 1.0 for profile in profiles)
    assert "4 to 7 words" in profiles[2].phrase_guidance
    assert "3 to 5 words" in profiles[-1].phrase_guidance


def test_natural_voice_formula_enforces_us_english_and_human_style() -> None:
    profile = VoiceDesigner.build_profile(
        age=85,
        gender="female",
        language="en-US",
        emotion="calm",
        description="Soft and sincere.",
        seed=2025,
        candidate_index=2,
    )
    prompt = profile.effective_description
    assert "85-year-old American woman" in prompt
    assert "General American English" in prompt
    assert "AI assistant" in prompt
    assert "not by stretching vowels" in prompt
    assert "Soft and sincere" in prompt


def test_candidate_profiles_change_with_candidate_index() -> None:
    first = VoiceDesigner.build_profile(
        age=72,
        gender="male",
        language="en-US",
        emotion="warm",
        description="",
        seed=2025,
        candidate_index=0,
    )
    second = VoiceDesigner.build_profile(
        age=72,
        gender="male",
        language="en-US",
        emotion="warm",
        description="",
        seed=2025,
        candidate_index=1,
    )
    assert first.identity_traits != second.identity_traits


def test_voice_designer_reports_missing_environment(tmp_path: Path) -> None:
    designer = VoiceDesigner(
        tmp_path / "voices",
        tmp_path / "candidates",
        python_executable=str(tmp_path / "missing-python"),
        worker_path=tmp_path / "missing-worker.py",
    )
    try:
        designer.generate_candidates(
            name="Test",
            description="A natural adult speaker with a clear voice.",
            sample_text="Testing.",
            seed=1,
        )
    except RuntimeError as error:
        assert "isolated MOSS VoiceGenerator environment is missing" in str(error)
    else:
        raise AssertionError("Expected a missing environment error")


def test_voice_designer_pins_verified_moss_snapshot(tmp_path: Path) -> None:
    designer = VoiceDesigner(
        tmp_path / "voices",
        tmp_path / "candidates",
        python_executable=sys.executable,
        worker_path=tmp_path / "worker.py",
        model_cache_dir=tmp_path / "models",
    )
    assert designer.MODEL_REVISION == "97521ec"
    assert designer.model_cache_dir == tmp_path / "models"


def test_voice_worker_uses_verified_moss_snapshot_and_official_api() -> None:
    worker_source = (Path(__file__).resolve().parents[1] / "voice_worker.py").read_text(encoding="utf-8")
    assert "OpenMOSS-Team/MOSS-VoiceGenerator" not in worker_source  # model id arrives from VoiceDesigner
    assert "snapshot_download" in worker_source
    assert "resolve_verified_model_snapshot" in worker_source
    assert "AutoProcessor.from_pretrained" in worker_source
    assert "audio_temperature" in worker_source
    assert "evaluate_acoustic_quality" in worker_source
