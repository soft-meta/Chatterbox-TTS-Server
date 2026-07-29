from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from threading import RLock

from utils import safe_filename


class VoiceDesigner:
    """Generate a short text-described reference voice in an isolated process.

    Chatterbox 0.1.7 and Parler-TTS 0.2.3 require incompatible Transformers
    versions. Running Parler-TTS in a separate Python environment prevents one
    engine from replacing the dependencies of the other.
    """

    MODEL_ID = "parler-tts/parler-tts-mini-v1.1"

    def __init__(
        self,
        output_dir: Path,
        device: str = "cuda",
        *,
        python_executable: str | None = None,
        worker_path: Path | None = None,
        timeout_seconds: int = 1800,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.python_executable = (
            python_executable
            or os.getenv("SOFTMETA_VOICE_PYTHON")
            or sys.executable
        )
        self.worker_path = worker_path or Path(__file__).with_name("voice_worker.py")
        self.timeout_seconds = timeout_seconds
        self._lock = RLock()

    @staticmethod
    def _complete_description(description: str) -> str:
        text = " ".join(description.strip().split())
        quality = (
            " The speaker uses natural American English pronunciation, realistic human timing, "
            "subtle breath and emotion, very clear close-microphone audio, and almost no background noise."
        )
        return text if "very clear" in text.lower() else text + quality

    def _next_output_path(self, name: str, seed: int) -> Path:
        stem = safe_filename(name, "Generated_Voice")
        candidate = self.output_dir / f"{stem}_{seed}.wav"
        counter = 2
        while candidate.exists():
            candidate = self.output_dir / f"{stem}_{seed}_{counter}.wav"
            counter += 1
        return candidate

    def _validate_runner(self) -> None:
        python_path = Path(self.python_executable)
        if not python_path.exists():
            raise RuntimeError(
                "The isolated Generate Voice environment is missing. "
                "Run the SoftMeta Colab installation cell again from a fresh runtime. "
                f"Expected Python executable: {python_path}"
            )
        if not self.worker_path.exists():
            raise RuntimeError(f"Generate Voice worker was not found: {self.worker_path}")

    def generate(self, *, name: str, description: str, sample_text: str, seed: int) -> Path:
        with self._lock:
            self._validate_runner()
            output_path = self._next_output_path(name, seed)
            request = {
                "model_id": self.MODEL_ID,
                "description": self._complete_description(description),
                "sample_text": sample_text.strip(),
                "seed": int(seed),
                "device": self.device,
                "output_path": str(output_path),
            }

            request_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    prefix="softmeta_voice_",
                    delete=False,
                    encoding="utf-8",
                ) as handle:
                    json.dump(request, handle, ensure_ascii=False)
                    request_path = Path(handle.name)

                env = {
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                }
                result = subprocess.run(
                    [
                        self.python_executable,
                        "-u",
                        str(self.worker_path),
                        "--request",
                        str(request_path),
                    ],
                    cwd=str(self.worker_path.parent),
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout_seconds,
                    check=False,
                )

                if result.returncode != 0:
                    output_path.unlink(missing_ok=True)
                    details = (result.stdout or "No worker output.")[-8000:]
                    raise RuntimeError(
                        "The isolated Generate Voice worker failed.\n"
                        f"Python: {self.python_executable}\n"
                        f"Exit code: {result.returncode}\n"
                        f"Worker output:\n{details}"
                    )
                if not output_path.exists() or output_path.stat().st_size < 1000:
                    output_path.unlink(missing_ok=True)
                    details = (result.stdout or "No worker output.")[-4000:]
                    raise RuntimeError(
                        "Generate Voice finished without creating a valid WAV file.\n"
                        f"Worker output:\n{details}"
                    )
                return output_path
            except subprocess.TimeoutExpired as error:
                output_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Generate Voice exceeded the {self.timeout_seconds // 60}-minute timeout."
                ) from error
            finally:
                if request_path is not None:
                    request_path.unlink(missing_ok=True)
