from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GenerationOptions(BaseModel):
    model: Literal[
        "chatterbox",
        "chatterbox-turbo",
        "chatterbox-nano",
        "chatterbox-multilingual",
    ] = "chatterbox"
    language: str = "en"
    temperature: float = Field(0.8, ge=0.05, le=2.0)
    exaggeration: float = Field(0.65, ge=0.0, le=2.0)
    cfg_weight: float = Field(0.35, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.2, ge=1.0, le=3.0)
    min_p: float = Field(0.05, ge=0.0, le=1.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    top_k: int = Field(1000, ge=1, le=5000)
    speed_factor: float = Field(1.0, ge=0.5, le=2.0)
    seed: int = Field(2025, ge=0, le=2_147_483_647)
    split_text: bool = True
    chunk_words: int = Field(90, ge=25, le=250)


class AudioJobCreate(BaseModel):
    audio_number: int = Field(1, ge=1, le=5)
    title: str = Field("", max_length=180)
    text: str = Field(..., min_length=1)
    voice_mode: Literal["predefined", "clone", "default"] = "default"
    voice_filename: str | None = None
    options: GenerationOptions = Field(default_factory=GenerationOptions)

    @model_validator(mode="after")
    def validate_voice(self) -> "AudioJobCreate":
        if self.voice_mode in {"predefined", "clone"} and not self.voice_filename:
            raise ValueError("A voice file is required for the selected voice mode.")
        return self


class GenerateAllRequest(BaseModel):
    jobs: list[AudioJobCreate] = Field(..., min_length=1, max_length=5)


class CutRequest(BaseModel):
    start_seconds: float = Field(0.0, ge=0.0)
    end_seconds: float | None = Field(None, gt=0.0)
    filename_prefix: str = "Selected"


class ModelLoadRequest(BaseModel):
    model: Literal[
        "chatterbox",
        "chatterbox-turbo",
        "chatterbox-nano",
        "chatterbox-multilingual",
    ]


class OpenAITTSRequest(BaseModel):
    model: str = "chatterbox"
    input: str
    voice: str | None = None
    response_format: Literal["wav"] = "wav"
    speed: float = Field(1.0, ge=0.5, le=2.0)
