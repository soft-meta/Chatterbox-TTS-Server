from __future__ import annotations

import re
from pathlib import Path

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\b[\w’'-]+\b", flags=re.UNICODE)


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def split_text(text: str, max_words: int = 90) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(cleaned) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        words = sentence.split()
        if len(words) > max_words:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_words = 0
            for index in range(0, len(words), max_words):
                chunks.append(" ".join(words[index:index + max_words]))
            continue
        if current and current_words + len(words) > max_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def safe_filename(value: str, fallback: str = "Untitled_Audio") -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_-")
    return value[:120] or fallback


def resolve_inside(root: Path, filename: str) -> Path:
    candidate = (root / Path(filename).name).resolve()
    root_resolved = root.resolve()
    if root_resolved not in candidate.parents:
        raise ValueError("Invalid filename.")
    return candidate


def estimated_audio_seconds(words: int, speed_factor: float = 1.0, wpm: float = 145.0) -> float:
    if words <= 0:
        return 0.0
    return (words / wpm) * 60.0 / max(speed_factor, 0.1)
