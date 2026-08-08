from __future__ import annotations

import re
from pathlib import Path

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+")
_CLAUSE_BOUNDARY = re.compile(r"(?<=[,;:])\s+")
_WORD = re.compile(r"\b[\w’'-]+\b", flags=re.UNICODE)
_SAFE_TURBO_TAG = re.compile(r"\[(?:happy|surprised|dramatic|narration)\]", flags=re.IGNORECASE)


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def prepare_american_english_tts_text(text: str) -> str:
    """Normalize punctuation for stable English Turbo narration without changing wording.

    This deliberately does *not* pretend to change a cloned speaker's accent. It keeps
    the input in English and removes punctuation patterns that can cause Turbo to hold
    very long pauses, while preserving the supported expression tags inserted by the
    Serious Senior Advisor director.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    protected: list[str] = []

    def protect_tag(match: re.Match[str]) -> str:
        protected.append(match.group(0).lower())
        return f"__SMTAG_{len(protected) - 1}__"

    normalized = _SAFE_TURBO_TAG.sub(protect_tag, normalized)

    # Markdown heading markers are visual, not spoken content.
    normalized = re.sub(r"(?m)^\s*#{1,6}\s*", "", normalized)

    # Parenthetical/bracket characters can produce exaggerated pauses in Turbo.
    # Keep the words, but turn round-parenthetical boundaries into a light comma.
    normalized = re.sub(r"[ \t]*\([ \t]*", ", ", normalized)
    normalized = re.sub(r"[ \t]*\)[ \t]*", ", ", normalized)
    normalized = re.sub(r"[\[\]{}]", " ", normalized)

    # Prefer simple American-English narration punctuation for the TTS tokenizer.
    normalized = normalized.replace("—", ", ").replace("–", ", ")
    normalized = re.sub(r"\s+-{2,}\s+", ", ", normalized)
    normalized = re.sub(r"\.{3,}", ".", normalized)
    normalized = re.sub(r"[ \t]*[;:][ \t]*", ", ", normalized)

    # Common title abbreviations can otherwise be mistaken for sentence endings.
    abbreviation_map = {
        r"\bDr\.\s*": "Doctor ",
        r"\bMr\.\s*": "Mister ",
        r"\bMrs\.\s*": "Missus ",
        r"\bMs\.\s*": "Ms ",
    }
    for pattern, replacement in abbreviation_map.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"[ \t]*,[ \t]*,+", ", ", normalized)
    normalized = re.sub(r"[ \t]+([,.!?])", r"\1", normalized)
    normalized = re.sub(r",[ \t]*([.!?])", r"\1", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()

    for index, tag in enumerate(protected):
        normalized = normalized.replace(f"__SMTAG_{index}__", tag)
    return normalized


def split_text(text: str, max_words: int = 90, *, prefer_clauses: bool = False) -> list[str]:
    """Sentence-aware long-text splitter with safe fallback for long sentences."""
    cleaned = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n"))
    cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()
    if not cleaned:
        return []

    sentences: list[str] = []
    for paragraph in cleaned.split("\n"):
        sentences.extend(part.strip() for part in _SENTENCE_BOUNDARY.split(paragraph) if part.strip())

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            chunks.append(" ".join(current).strip())
            current = []
            current_words = 0

    for sentence in sentences:
        words = sentence.split()
        if len(words) > max_words:
            flush()
            if not prefer_clauses:
                for index in range(0, len(words), max_words):
                    chunks.append(" ".join(words[index:index + max_words]))
                continue

            clauses = [part.strip() for part in _CLAUSE_BOUNDARY.split(sentence) if part.strip()]
            clause_chunk: list[str] = []
            clause_words = 0

            def flush_clause_chunk() -> None:
                nonlocal clause_chunk, clause_words
                if clause_chunk:
                    chunks.append(" ".join(clause_chunk).strip())
                    clause_chunk = []
                    clause_words = 0

            for clause in clauses:
                clause_parts = clause.split()
                if len(clause_parts) > max_words:
                    flush_clause_chunk()
                    for index in range(0, len(clause_parts), max_words):
                        chunks.append(" ".join(clause_parts[index:index + max_words]))
                    continue
                if clause_chunk and clause_words + len(clause_parts) > max_words:
                    flush_clause_chunk()
                clause_chunk.append(clause)
                clause_words += len(clause_parts)
            flush_clause_chunk()
            continue
        if current and current_words + len(words) > max_words:
            flush()
        current.append(sentence)
        current_words += len(words)
    flush()
    return [chunk for chunk in chunks if chunk]


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
