from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from softmeta_chatterbox import GenerationSettings, SoftMetaChatterboxEngine
except ImportError as error:  # pragma: no cover
    raise RuntimeError(
        "softmeta-chatterbox-v2 is not installed. Install soft-meta/chatterbox-v2 first."
    ) from error


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _turbo_creator_sampling(
    temperature: float,
    top_p: float,
    *,
    exaggeration: float,
    cfg_weight: float,
) -> tuple[float, float]:
    """Make the creator Exaggeration/CFG sliders useful on Turbo.

    Current official Chatterbox Turbo exposes these arguments but warns that its
    inference path ignores native CFG/exaggeration.  We therefore keep 0.50 as a
    neutral Exaggeration point and bridge the creator controls conservatively into
    Turbo's supported sampling controls.  This changes delivery variation without
    adding text rewriting, emotion tags, retries, QC or any speech post-processing.
    """
    base_temperature = _clamp(temperature, 0.05, 2.0)
    base_top_p = _clamp(top_p, 0.10, 1.0)
    expression = _clamp(exaggeration, 0.0, 2.0) - 0.5
    guidance = _clamp(cfg_weight, 0.0, 1.0)

    # Higher Exaggeration opens sampling slightly; higher CFG-like guidance pulls
    # it back toward a steadier, more literal delivery.  The ranges are deliberately
    # small so senior-advisor clarity and word fidelity remain the priority.
    effective_temperature = _clamp(
        base_temperature + (0.12 * expression) - (0.06 * guidance), 0.05, 2.0
    )
    # Keep Top P at the model's stable hidden default. The creator asked to remove
    # Top P as a tuning dimension, so Exaggeration/CFG only nudge Temperature.
    effective_top_p = base_top_p
    return round(effective_temperature, 4), round(effective_top_p, 4)


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
        requested_exaggeration = float(options.get("exaggeration", 0.5))
        requested_cfg = float(options.get("cfg_weight", 0.0))
        effective_temperature, effective_top_p = _turbo_creator_sampling(
            float(options["temperature"]),
            float(options["top_p"]),
            exaggeration=requested_exaggeration,
            cfg_weight=requested_cfg,
        )
        settings = GenerationSettings(
            temperature=effective_temperature,
            # Preserve Exaggeration in reference preparation.  The pinned adapter
            # strips native Turbo exaggeration/CFG from model.generate(), avoiding
            # unsupported-parameter warnings while the sampling bridge above makes
            # both creator sliders materially affect Turbo delivery.
            exaggeration=requested_exaggeration,
            cfg_weight=requested_cfg,
            repetition_penalty=float(options["repetition_penalty"]),
            min_p=float(options["min_p"]),
            top_p=effective_top_p,
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

        return EngineResult(waveform=waveform, sample_rate=sample_rate)
