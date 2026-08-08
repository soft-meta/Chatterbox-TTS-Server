from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio.functional as audio_functional

try:
    from softmeta_chatterbox import GenerationSettings, SoftMetaChatterboxEngine
except ImportError as error:  # pragma: no cover
    raise RuntimeError(
        "softmeta-chatterbox-v2 is not installed. Install soft-meta/chatterbox-v2 first."
    ) from error


@dataclass(slots=True)
class EngineResult:
    waveform: torch.Tensor
    sample_rate: int


class EngineService:
    def __init__(self, device: str = "auto") -> None:
        self.runtime = SoftMetaChatterboxEngine(device=device)
        # L4/Ada GPUs benefit from TensorFloat-32 for transformer matmuls.
        # This changes compute precision, not audio sample rate or playback speed.
        if self.runtime.device == "cuda" and torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @property
    def loaded_model(self) -> str | None:
        return self.runtime.model_name

    @property
    def device(self) -> str:
        return self.runtime.device

    def status(self) -> dict[str, Any]:
        return self.runtime.status_dict()

    def load(self, model_name: str) -> None:
        self.runtime.load(model_name)  # type: ignore[arg-type]

    def unload(self) -> None:
        self.runtime.unload()

    def generate(
        self,
        text: str,
        *,
        model_name: str,
        reference_audio: Path | None,
        language: str,
        options: dict[str, Any],
    ) -> EngineResult:
        settings = GenerationSettings(
            temperature=float(options["temperature"]),
            exaggeration=float(options["exaggeration"]),
            cfg_weight=float(options["cfg_weight"]),
            repetition_penalty=float(options["repetition_penalty"]),
            min_p=float(options["min_p"]),
            top_p=float(options["top_p"]),
            top_k=int(options["top_k"]),
            seed=int(options["seed"]),
        )
        # No gradients are needed for TTS. inference_mode lowers Python/autograd
        # overhead and memory pressure during long-form generation.
        with torch.inference_mode():
            waveform, sample_rate = self.runtime.generate(
                text,
                model_name=model_name,  # type: ignore[arg-type]
                reference_audio=reference_audio,
                language=language,
                settings=settings,
            )

        speed = float(options.get("speed_factor", 1.0))
        if abs(speed - 1.0) > 0.001:
            # Change the sample count, then save at the original sample rate. This
            # changes playback speed without adding a heavyweight DSP dependency.
            resample_rate = max(4000, int(sample_rate / speed))
            waveform = audio_functional.resample(waveform, sample_rate, resample_rate)
        return EngineResult(waveform=waveform, sample_rate=sample_rate)
