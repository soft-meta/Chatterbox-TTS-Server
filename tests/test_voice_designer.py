from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from voice_designer import VoiceDesigner


def test_voice_designer_uses_isolated_runner(tmp_path: Path) -> None:
    worker = tmp_path / "fake_worker.py"
    worker.write_text(
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
    designer = VoiceDesigner(
        tmp_path / "voices",
        device="cpu",
        python_executable=sys.executable,
        worker_path=worker,
        timeout_seconds=30,
    )
    output = designer.generate(
        name="Test Voice",
        description="A natural adult American speaker with a calm delivery.",
        sample_text="This is a short test.",
        seed=2025,
    )
    assert output.exists()
    assert output.name.startswith("Test_Voice_2025")
    assert sf.info(output).samplerate == 24000


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
