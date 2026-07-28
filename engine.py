from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio.functional as AF

try:
    from softmeta_chatterbox import GenerationSettings, SoftMetaChatterboxEngine
except ImportError as error:  # pragma: no cover - startup guidance
    raise RuntimeError(
        "softmeta-chatterbox-v2 is not installed. Install the soft-meta/chatterbox-v2 repository first."
    ) from error


@dataclass(slots=True)
class EngineResult:
    waveform: torch.Tensor
    sample_rate: int


class EngineService:
    def __init__(self, device: str = "auto") -> None:
        self.runtime = SoftMetaChatterboxEngine(device=device)

    @property
    def loaded_model(self) -> str | None:
        return self.runtime.model_name

    @property
    def device(self) -> str:
        return self.runtime.device

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
        options: dict,
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
        waveform, sample_rate = self.runtime.generate(
            text,
            model_name=model_name,  # type: ignore[arg-type]
            reference_audio=reference_audio,
            language=language,
            settings=settings,
        )
        speed = float(options.get("speed_factor", 1.0))
        if abs(speed - 1.0) > 0.001:
            target_rate = max(4000, int(sample_rate / speed))
            waveform = AF.resample(waveform, sample_rate, target_rate)
        return EngineResult(waveform=waveform, sample_rate=sample_rate)
