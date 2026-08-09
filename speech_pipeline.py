from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from utils import count_words, prepare_american_english_tts_text, split_text

_TURBO_TAG = re.compile(r"\[(?:happy|surprised|dramatic|narration)\]", re.IGNORECASE)
_EMOTION_TAG_CAPTURE = re.compile(r"^\s*\[(happy|surprised|dramatic|narration)\]\s*", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+")

# Deliberately serious. These are comprehension/emphasis cues, not entertainment cues.
_STRONG_EMPHASIS = re.compile(
    r"\b(?:do not ignore|never ignore|call 911|call emergency services|medical emergency|"
    r"seek emergency care|could save your life|may save your life|life[- ]threatening|"
    r"serious warning|warning sign)\b",
    re.IGNORECASE,
)
_CALM_EMPHASIS = re.compile(
    r"\b(?:remember this|keep this in mind|this is important|the important thing|"
    r"what matters most|what matters|the biggest mistake|one of the biggest mistakes|"
    r"most people (?:do not|don't) realize|pay attention|now pay attention|"
    r"the key is|the key point|this matters)\b",
    re.IGNORECASE,
)
_CTA = re.compile(
    r"\b(?:subscribe|follow (?:me|us|this page)|like (?:this|the) video|share (?:this|the) video|comment below)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class KeyEmphasisAnalysis:
    placements: list[dict[str, Any]]


def key_emphasis_for_sentence(sentence: str) -> tuple[str | None, float]:
    clean = _TURBO_TAG.sub("", sentence).strip()
    words = count_words(clean)
    if words < 5 or words > 55 or _CTA.search(clean):
        return None, 0.0
    if _STRONG_EMPHASIS.search(clean):
        return "dramatic", 5.0
    if _CALM_EMPHASIS.search(clean):
        return "narration", 3.1
    return None, 0.0


def analyze_key_emphasis(text: str) -> KeyEmphasisAnalysis:
    clean = text.replace("\r\n", "\n").replace("\r", "\n")
    sentences: list[tuple[str, int]] = []
    cursor = 0
    for line in clean.splitlines():
        for part in _SENTENCE.split(line.strip()):
            part = part.strip()
            if not part:
                continue
            sentences.append((part, cursor))
            cursor += count_words(_TURBO_TAG.sub("", part))

    candidates: list[dict[str, Any]] = []
    for sentence, start_word in sentences:
        tag, score = key_emphasis_for_sentence(sentence)
        if tag:
            candidates.append({
                "kind": "serious-emphasis",
                "tag": tag,
                "score": score,
                "word_position": start_word,
                "excerpt": sentence[:140],
            })

    # Sparse by design: roughly one emphasis moment per 180 spoken words, max 3.
    total_words = max(cursor, 1)
    max_items = min(3, max(1, (total_words + 179) // 180))
    selected: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda x: (-float(x["score"]), int(x["word_position"]))):
        if len(selected) >= max_items:
            break
        pos = int(item["word_position"])
        if any(abs(pos - int(chosen["word_position"])) < 70 for chosen in selected):
            continue
        selected.append(item)
    selected.sort(key=lambda x: int(x["word_position"]))
    return KeyEmphasisAnalysis(placements=selected)


def _soften_long_sentence(sentence: str, max_phrase_words: int = 30) -> str:
    """Add only punctuation-level breathing help; never rewrite the speaker's words."""
    if count_words(_TURBO_TAG.sub("", sentence)) <= max_phrase_words:
        return sentence.strip()

    # Prefer discourse conjunctions in the middle of a long sentence. We insert a
    # comma only where one is absent. This gives Turbo a phrasing cue without adding
    # or removing spoken words.
    words = sentence.split()
    if len(words) <= max_phrase_words:
        return sentence.strip()
    lo = max(8, len(words) // 3)
    hi = min(len(words) - 7, (len(words) * 2) // 3)
    choices = {"but", "because", "while", "although", "however", "so", "which", "when", "yet"}
    target = len(words) // 2
    positions = [i for i in range(lo, hi + 1) if re.sub(r"[^A-Za-z]", "", words[i]).lower() in choices]
    if not positions:
        return sentence.strip()
    idx = min(positions, key=lambda i: abs(i - target))
    if idx > 0 and not words[idx - 1].endswith((",", ";", ":")):
        words[idx - 1] = words[idx - 1].rstrip() + ","
    return " ".join(words).strip()


def _looks_like_structural_heading(line: str) -> bool:
    clean = _TURBO_TAG.sub("", line).strip()
    if not clean:
        return False
    words = clean.split()
    if re.match(
        r"^(?:#{1,6}\s+|(?:number\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)|"
        r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d{1,2})\s*[.):-])",
        clean, flags=re.IGNORECASE,
    ) and len(words) <= 20:
        return True
    if clean.endswith(":") and len(words) <= 12 and not any(ch in clean for ch in "?!"):
        return True
    letters = "".join(ch for ch in clean if ch.isalpha())
    if letters and letters.upper() == letters and len(words) <= 12:
        return True
    alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
    if alpha_words and len(words) <= 12 and not clean.endswith((".", "?", "!")):
        caps = sum(1 for w in alpha_words if next((c for c in w if c.isalpha()), "").isupper())
        if caps / len(alpha_words) >= 0.60:
            return True
    return False


def prepare_senior_clear_speech_text(text: str) -> str:
    """Prepare calm, intelligible American-English narration without changing wording.

    The function is intentionally conservative: stable punctuation, common spoken
    title expansion, and a single phrasing cue for unusually long sentences. It does
    not claim to override the accent carried by a clone reference.
    """
    out_lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            out_lines.append("")
            continue
        # Preserve structural heading punctuation until the long-form section planner
        # turns it into a deliberate spoken heading pattern. Generic normalization
        # would otherwise turn the colon into a comma and lose section detection.
        if _looks_like_structural_heading(stripped):
            out_lines.append(stripped)
            continue
        normalized = prepare_american_english_tts_text(stripped)
        parts = [p.strip() for p in _SENTENCE.split(normalized) if p.strip()]
        out_lines.append(" ".join(_soften_long_sentence(part) for part in parts))
    return "\n".join(out_lines).strip()


PACE_PROFILES: dict[str, dict[str, int]] = {
    # Engineering listening profiles rather than medical claims. Clear articulation
    # remains primary; these bands simply prevent an elderly-advice narrator from
    # rushing or being mechanically forced to one exact WPM.
    "60s": {"body": 154, "band": 8, "intro_delta": 6, "important_delta": -7, "section_delta": -3, "reset_delta": 3},
    "70s": {"body": 148, "band": 8, "intro_delta": 6, "important_delta": -7, "section_delta": -3, "reset_delta": 3},
    "80s": {"body": 142, "band": 8, "intro_delta": 6, "important_delta": -7, "section_delta": -3, "reset_delta": 3},
    "90plus": {"body": 136, "band": 8, "intro_delta": 6, "important_delta": -7, "section_delta": -3, "reset_delta": 3},
}


def pace_profile(name: str) -> dict[str, int]:
    return PACE_PROFILES.get(name, PACE_PROFILES["70s"])


@dataclass(slots=True)
class SpeechSegment:
    text: str
    role: str
    word_count: int
    start_word: int
    end_word: int
    pause_before_ms: int
    target_wpm: int
    min_wpm: int
    max_wpm: int
    importance: int = 0
    retention_reset: bool = False
    emotion_tag: str | None = None




def emotion_tag_for_text(text: str) -> str | None:
    """Return a supported auto-emotion tag only when it starts this speech span."""
    match = _EMOTION_TAG_CAPTURE.match(text or "")
    return match.group(1).lower() if match else None


def _emotion_aware_chunks(text: str, max_words: int) -> list[tuple[str, str | None]]:
    """Pack neutral narration normally while isolating tagged emotion sentences.

    Auto Emotion places control tags immediately before a sentence. Letting that
    sentence disappear inside an 80+ word chunk makes Original's chunk-wide controls
    effectively non-local and makes Turbo's native tag easy to wash out in long-form
    prosody. This packer keeps each tagged sentence in a compact 1–2 sentence span
    while normal narration retains the efficient long-form chunk size.
    """
    cleaned = re.sub(r"[ \t]+", " ", text.strip())
    if not cleaned:
        return []
    sentences = [part.strip() for part in _SENTENCE.split(cleaned) if part.strip()]
    if not sentences:
        return [(cleaned, emotion_tag_for_text(cleaned))]

    packed: list[tuple[str, str | None]] = []
    neutral: list[str] = []
    neutral_words = 0

    def flush_neutral() -> None:
        nonlocal neutral, neutral_words
        if not neutral:
            return
        value = " ".join(neutral).strip()
        if value:
            # Merge only with another neutral span. Never merge neutral speech into
            # an emotion span because Original applies its expression per model call.
            if packed and packed[-1][1] is None and count_words(value) < 8 and count_words(packed[-1][0]) + count_words(value) <= max_words:
                packed[-1] = (f"{packed[-1][0]} {value}".strip(), None)
            else:
                packed.append((value, None))
        neutral = []
        neutral_words = 0

    i = 0
    emotion_limit = max(18, min(44, max_words))
    while i < len(sentences):
        sentence = sentences[i]
        tag = emotion_tag_for_text(sentence)
        sentence_words = count_words(sentence)
        if tag:
            flush_neutral()
            expressive = sentence
            expressive_words = sentence_words
            # A very short cue can sound abrupt as a standalone model call. Add one
            # following neutral sentence only when the compact emotion span stays small.
            if expressive_words < 12 and i + 1 < len(sentences):
                following = sentences[i + 1]
                if emotion_tag_for_text(following) is None:
                    combined_words = expressive_words + count_words(following)
                    if combined_words <= emotion_limit:
                        expressive = f"{expressive} {following}".strip()
                        expressive_words = combined_words
                        i += 1
            if expressive_words <= emotion_limit:
                packed.append((expressive, tag))
            else:
                pieces = split_text(expressive, emotion_limit, prefer_clauses=True)
                for piece_index, piece in enumerate(pieces):
                    packed.append((piece, tag if piece_index == 0 else None))
            i += 1
            continue

        if sentence_words > max_words:
            flush_neutral()
            for piece in split_text(sentence, max_words, prefer_clauses=True):
                packed.append((piece, None))
            i += 1
            continue
        if neutral and neutral_words + sentence_words > max_words:
            flush_neutral()
        neutral.append(sentence)
        neutral_words += sentence_words
        i += 1

    flush_neutral()
    return [(value, tag) for value, tag in packed if value.strip()]

def _heading_for_speech(line: str) -> str:
    clean = _TURBO_TAG.sub("", line).strip()
    # Numbered headings sound clearer when the ordinal and title are separated by
    # a full stop instead of a colon/dash. The visible UI text remains unchanged.
    m = re.match(
        r"^\s*((?:number\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)|"
        r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d{1,2}))\s*[.):-]\s*(.+)$",
        clean,
        flags=re.IGNORECASE,
    )
    if m:
        prefix, title = m.group(1).strip(), m.group(2).strip().rstrip(".:;!?")
        return f"{prefix}. {title}."
    return clean if clean.endswith((".", "?", "!")) else clean + "."


def build_long_form_segments(
    text: str,
    *,
    max_words: int = 85,
    heading_detector=None,
    retention_positions: list[int] | None = None,
    age_profile: str = "70s",
) -> list[SpeechSegment]:
    """Create long-form TTS segments while preserving section structure.

    The first segment is kept near a 30-second intro window. Numbered headings begin
    new sections but stay attached to their first body sentences so we do not waste a
    model pass on tiny heading-only chunks.
    """
    from utils import split_text

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    detector = heading_detector or (lambda _line: False)
    retention_positions = sorted(retention_positions or [])
    pace = pace_profile(age_profile)

    blocks: list[tuple[bool, list[str]]] = []
    current: list[str] = []
    current_heading = False
    for raw in normalized.splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = bool(detector(line))
        if heading and current:
            blocks.append((current_heading, current))
            current = []
        if heading:
            current_heading = True
            current.append(_heading_for_speech(line))
        else:
            if not current:
                current_heading = False
            current.append(line)
    if current:
        blocks.append((current_heading, current))

    segments: list[SpeechSegment] = []
    word_cursor = 0
    for block_index, (starts_with_heading, lines) in enumerate(blocks):
        block_text = " ".join(lines).strip()
        # Keep the first generated pass around 25-30 seconds at the intended intro
        # pace. Later chunks use the user's long-form chunk size.
        first_limit = min(max_words, 72) if not segments else max_words
        raw_chunks = _emotion_aware_chunks(block_text, first_limit)
        if not raw_chunks:
            continue
        for chunk_index, (chunk, emotion_tag) in enumerate(raw_chunks):
            wc = count_words(chunk)
            start, end = word_cursor, word_cursor + wc
            has_retention_reset = any(start <= pos < end for pos in retention_positions)
            importance = len(analyze_key_emphasis(chunk).placements)

            if not segments:
                role = "intro"
            elif starts_with_heading and chunk_index == 0:
                role = "section"
            elif emotion_tag:
                role = "emotion"
            elif has_retention_reset:
                role = "reset"
            else:
                role = "body"

            body_target = int(pace["body"])
            band = int(pace["band"])
            if role == "intro":
                target_wpm = body_target + int(pace["intro_delta"])
                pause_before = 0
            elif role == "emotion":
                emotion_delta = {"narration": -4, "happy": 0, "surprised": -2, "dramatic": -6}.get(emotion_tag or "", -3)
                target_wpm = body_target + emotion_delta
                # Give native/model-controlled expression room to breathe. A wider
                # pace band prevents atempo from flattening an intentional prosody cue.
                band = max(band, 13)
                pause_before = 150 if emotion_tag == "dramatic" else 115
            elif importance:
                target_wpm = body_target + int(pace["important_delta"])
                pause_before = 300 if role != "section" else 520
            elif role == "section":
                target_wpm = body_target + int(pace["section_delta"])
                pause_before = 520
            elif role == "reset":
                target_wpm = body_target + int(pace["reset_delta"])
                pause_before = 360
            else:
                target_wpm = body_target
                pause_before = 120 if segments and segments[-1].emotion_tag else 190

            segments.append(
                SpeechSegment(
                    text=chunk,
                    role=role,
                    word_count=wc,
                    start_word=start,
                    end_word=end,
                    pause_before_ms=pause_before,
                    target_wpm=target_wpm,
                    min_wpm=target_wpm - band,
                    max_wpm=target_wpm + band,
                    importance=importance,
                    retention_reset=has_retention_reset,
                    emotion_tag=emotion_tag,
                )
            )
            word_cursor = end
    return segments
