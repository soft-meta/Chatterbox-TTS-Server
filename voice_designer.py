from __future__ import annotations

import gc
import random
from pathlib import Path
from threading import RLock

import numpy as np
import soundfile as sf
import torch

from utils import safe_filename


class VoiceDesigner:
    """Lazy text-described voice sample generator using Parler-TTS Mini v1.1.

    The generated WAV is a reference sample. SoftMeta then uses that sample through
    Chatterbox zero-shot voice cloning for long-form audio generation.
    """

    MODEL_ID = "parler-tts/parler-tts-mini-v1.1"

    def __init__(self, output_dir: Path, device: str = "cuda") -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self._lock = RLock()

    @staticmethod
    def _complete_description(description: str) -> str:
        text = " ".join(description.strip().split())
        quality = (
            " The speaker uses natural American English pronunciation, realistic human timing, "
            "subtle breath and emotion, very clear close-microphone audio, and almost no background noise."
        )
        return text if "very clear" in text.lower() else text + quality

    def generate(self, *, name: str, description: str, sample_text: str, seed: int) -> Path:
        with self._lock:
            try:
                from parler_tts import ParlerTTSForConditionalGeneration
                from transformers import AutoTokenizer
            except ImportError as error:
                raise RuntimeError(
                    "Generate Voice requires parler-tts==0.2.3. Install the SoftMeta Colab requirements."
                ) from error

            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            model = None
            try:
                model = ParlerTTSForConditionalGeneration.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=dtype,
                ).to(self.device)
                prompt_tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
                description_tokenizer = AutoTokenizer.from_pretrained(
                    model.config.text_encoder._name_or_path
                )

                description_ids = description_tokenizer(
                    self._complete_description(description),
                    return_tensors="pt",
                ).input_ids.to(self.device)
                prompt_ids = prompt_tokenizer(
                    sample_text.strip(),
                    return_tensors="pt",
                ).input_ids.to(self.device)

                with torch.inference_mode():
                    generation = model.generate(
                        input_ids=description_ids,
                        prompt_input_ids=prompt_ids,
                        do_sample=True,
                    )

                audio = generation.detach().float().cpu().numpy().squeeze()
                if audio.ndim != 1 or audio.size < 1000:
                    raise RuntimeError("The voice designer returned an invalid audio sample.")

                stem = safe_filename(name, "Generated_Voice")
                filename = f"{stem}_{seed}.wav"
                path = self.output_dir / filename
                sf.write(path, audio, int(model.config.sampling_rate), subtype="PCM_16")
                return path
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
