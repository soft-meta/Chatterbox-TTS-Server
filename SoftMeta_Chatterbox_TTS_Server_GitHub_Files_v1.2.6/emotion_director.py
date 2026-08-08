from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from utils import count_words

# Chatterbox Turbo's native paralinguistic tokens. Auto-direction intentionally
# uses only a conservative subset suitable for serious senior-advice narration.
SUPPORTED_TURBO_TAGS = {
    "advertisement",
    "angry",
    "chuckle",
    "clear throat",
    "cough",
    "crying",
    "dramatic",
    "fear",
    "gasp",
    "groan",
    "happy",
    "laugh",
    "narration",
    "sarcastic",
    "shush",
    "sigh",
    "sniff",
    "surprised",
    "whispering",
}
AUTO_ALLOWED_TAGS = {"happy", "narration", "surprised", "dramatic"}
AUTO_FORBIDDEN_TAGS = {
    "angry", "laugh", "chuckle", "crying", "sarcastic", "fear", "gasp", "sigh",
    "groan", "shush", "sniff", "cough", "clear throat", "advertisement",
    "whispering",
}

_TAG_RE = re.compile(
    r"\[(?:" + "|".join(re.escape(tag) for tag in sorted(SUPPORTED_TURBO_TAGS, key=len, reverse=True)) + r")\]",
    flags=re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+")
_NUMBER_WORDS = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
_ORDINALS = "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth"
_HEADING_PREFIX = re.compile(
    rf"^\s*(?:#{1,6}\s+|(?:number\s+(?:\d+|{_NUMBER_WORDS})|{_ORDINALS}|\d{{1,2}})\s*[.):-])\s*",
    flags=re.IGNORECASE,
)
_CTA_RE = re.compile(
    r"\b(?:follow (?:me|us|this page)|subscribe|like (?:this|the) video|share (?:this|the) video|"
    r"leave a comment|comment below|click the link)\b",
    flags=re.IGNORECASE,
)

# High-precision phrase groups. The goal is not to label every emotion; it is to
# find a few moments where a subtle delivery change helps comprehension.
_PHRASES: dict[str, tuple[tuple[str, float], ...]] = {
    "happy": (
        ("the good news", 4.0), ("good news", 3.5), ("fortunately", 3.0),
        ("thankfully", 3.0), ("the encouraging part", 3.3), ("there is hope", 3.5),
        ("you can improve", 2.5), ("can improve", 2.1), ("you can change", 2.4),
        ("can change", 2.0), ("things got better", 3.0), ("began to improve", 2.7),
        ("started to improve", 2.7), ("felt better", 2.3), ("feel better", 2.0),
        ("made a real difference", 2.8), ("changed my life for the better", 3.4),
        ("I was grateful", 2.8), ("I am grateful", 2.8), ("happier", 2.0),
        ("relief", 1.9), ("hope", 1.5), ("better", 1.0),
        ("improve", 1.0), ("improved", 1.0), ("safer", 1.2),
        ("stronger", 1.0), ("easier", 0.9), ("healthy", 0.7),
        ("progress", 0.9), ("protect yourself", 1.2),
    ),
    "narration": (
        ("I wish I had", 4.0), ("wish I had", 3.5), ("I learned too late", 4.0),
        ("learned too late", 3.7), ("by the time I realized", 3.3),
        ("I regret", 3.7), ("my biggest regret", 4.0), ("unfortunately", 2.5),
        ("cost me years", 3.2), ("cost me", 2.0), ("years I cannot get back", 4.0),
        ("years I can't get back", 4.0), ("lost years", 3.1), ("I ignored", 2.2),
        ("I was lonely", 3.0), ("felt alone", 2.7), ("I suffered", 2.7),
        ("it hurt", 2.1), ("painful lesson", 3.0), ("hard lesson", 2.1),
        ("one of my mistakes", 2.5), ("one of the biggest mistakes", 3.4),
        ("mistake", 1.1), ("regret", 1.4), ("pain", 0.8), ("painful", 1.1),
        ("hurt", 0.9), ("lost", 0.7), ("lonely", 1.0), ("suffered", 1.0),
        ("suffering", 1.0), ("wrong", 0.7), ("ignored", 0.8),
        ("illness", 0.8), ("sick", 0.8), ("difficult", 0.7),
    ),
    "surprised": (
        ("I couldn't believe", 4.0), ("I could not believe", 4.0),
        ("what surprised me", 4.0), ("you may be surprised", 3.6),
        ("you might be surprised", 3.6), ("I was surprised", 3.6),
        ("to my surprise", 3.5), ("I never expected", 3.5),
        ("most people don't realize", 3.0), ("most people do not realize", 3.0),
        ("few people realize", 3.0), ("unexpected", 2.0), ("surprising", 2.3),
        ("I discovered", 1.8), ("I finally realized", 2.0), ("the truth is", 1.5),
        ("suddenly", 1.0), ("I did not expect", 2.3), ("I didn't expect", 2.3),
    ),
    "dramatic": (
        ("call 911", 5.0), ("call emergency services", 5.0), ("medical emergency", 4.7),
        ("seek emergency care", 4.7), ("could save your life", 4.5),
        ("may save your life", 4.3), ("life-threatening", 4.5),
        ("do not ignore", 3.6), ("never ignore", 3.8), ("warning sign", 3.2),
        ("serious warning", 3.6), ("urgent", 2.5),
    ),
}

_THRESHOLDS = {"happy": 2.2, "narration": 2.4, "surprised": 2.8, "dramatic": 4.0}
_LABELS = {"happy": "warm", "narration": "reflective", "surprised": "surprised", "dramatic": "serious"}


@dataclass(slots=True)
class _Sentence:
    line_index: int
    sentence_index: int
    text: str
    start_word: int
    end_word: int
    after_heading_gap: int | None


@dataclass(slots=True)
class _Candidate:
    sentence: _Sentence
    tag: str
    score: float


@dataclass(slots=True)
class EmotionAnalysis:
    tagged_text: str
    total_words: int
    applied_count: int
    protected_headings: int
    manual_tags: int
    by_tag: dict[str, int]
    mode: str = "Serious Senior Advisor"

    def public_summary(self, *, include_text: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "mode": self.mode,
            "total_words": self.total_words,
            "applied_count": self.applied_count,
            "protected_headings": self.protected_headings,
            "manual_tags": self.manual_tags,
            "by_tag": dict(self.by_tag),
            "labels": {_LABELS[tag]: count for tag, count in self.by_tag.items() if count},
        }
        if include_text:
            data["tagged_text"] = self.tagged_text
        return data


def strip_turbo_tags(text: str) -> str:
    return _TAG_RE.sub("", text)


def contains_turbo_tag(text: str) -> bool:
    return bool(_TAG_RE.search(text))


def is_heading(line: str) -> bool:
    clean = strip_turbo_tags(line).strip()
    if not clean:
        return False
    words = clean.split()
    if _HEADING_PREFIX.match(clean) and len(words) <= 20:
        return True
    if clean.startswith("#") and len(words) <= 20:
        return True
    # Short title lines ending in a colon are also treated as headings, but avoid
    # misclassifying ordinary spoken sentences containing multiple clauses.
    if clean.endswith(":") and len(words) <= 12 and clean.count(".") == 0 and clean.count("?") == 0:
        return True
    # Conservative ALL-CAPS title support.
    letters = "".join(ch for ch in clean if ch.isalpha())
    if letters and letters.upper() == letters and len(words) <= 12:
        return True
    # Short Title Case lines are common video/script headings. Require most words
    # to begin with capitals so normal spoken sentences are not swallowed.
    alpha_words = [word for word in words if any(ch.isalpha() for ch in word)]
    capitalized = sum(1 for word in alpha_words if next((ch for ch in word if ch.isalpha()), "").isupper())
    if (
        alpha_words
        and len(words) <= 12
        and not clean.endswith((".", "?", "!"))
        and capitalized / len(alpha_words) >= 0.60
    ):
        return True
    return False


def _sentence_parts(line: str) -> list[str]:
    clean = re.sub(r"[ \t]+", " ", line).strip()
    if not clean:
        return []
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(clean) if part.strip()]


def _score_sentence(text: str, previous: str, following: str) -> tuple[str | None, float]:
    if contains_turbo_tag(text) or _CTA_RE.search(text):
        return None, 0.0
    clean = strip_turbo_tags(text).strip()
    words = count_words(clean)
    if words < 5 or words > 70:
        return None, 0.0

    current = clean.lower()
    context_before = previous.lower()
    context_after = following.lower()
    scores = {tag: 0.0 for tag in AUTO_ALLOWED_TAGS}

    for tag, phrases in _PHRASES.items():
        for phrase, weight in phrases:
            if phrase.lower() in current:
                scores[tag] += weight
            elif phrase.lower() in context_before or phrase.lower() in context_after:
                # Neighbouring context helps disambiguate, but is intentionally weak
                # so the expressive cue still lands on the sentence carrying meaning.
                scores[tag] += weight * 0.18

    first_person = bool(re.search(r"\b(?:i|i'm|i've|my|me|we|our)\b", current))
    if first_person and scores["narration"] > 0:
        scores["narration"] += 0.45
    if "!" in clean:
        # Do not turn punctuation into excitement. It only slightly reinforces a
        # candidate already supported by meaning.
        for tag in ("happy", "surprised"):
            if scores[tag] > 0:
                scores[tag] += 0.15

    # Dramatic is deliberately hard to trigger; general negative wording must not
    # make an elderly advisor sound angry or theatrical.
    if scores["dramatic"] and not any(
        phrase in current
        for phrase in (
            "call 911", "emergency", "save your life", "life-threatening",
            "do not ignore", "never ignore", "serious warning",
        )
    ):
        scores["dramatic"] *= 0.65

    tag, score = max(scores.items(), key=lambda item: item[1])
    if score < _THRESHOLDS[tag]:
        return None, score
    return tag, score


def analyze_serious_senior_advisor(text: str) -> EmotionAnalysis:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    manual_tags = len(_TAG_RE.findall(normalized))
    # Serious Advisor mode owns expression direction. Remove any pasted Turbo tags
    # first so accidental [laugh]/[angry]/etc. cannot bypass the conservative policy.
    sanitized = strip_turbo_tags(normalized)
    lines = sanitized.split("\n")
    total_words = count_words(sanitized)
    protected_headings = 0

    # Pass 1: read the whole script, identify headings, sentence positions and
    # context. No tags are inserted yet.
    sentence_map: dict[int, list[_Sentence]] = {}
    word_cursor = 0
    last_heading_end: int | None = None
    for line_index, raw_line in enumerate(lines):
        if is_heading(raw_line):
            protected_headings += 1
            # Guarantee headings never contain Turbo tags, even if one was pasted
            # manually by mistake.
            heading_words = count_words(strip_turbo_tags(raw_line))
            word_cursor += heading_words
            last_heading_end = word_cursor
            continue
        parts = _sentence_parts(raw_line)
        sentence_list: list[_Sentence] = []
        for sentence_index, part in enumerate(parts):
            speech_words = count_words(strip_turbo_tags(part))
            start = word_cursor
            end = word_cursor + speech_words
            sentence_list.append(
                _Sentence(
                    line_index=line_index,
                    sentence_index=sentence_index,
                    text=part,
                    start_word=start,
                    end_word=end,
                    after_heading_gap=None if last_heading_end is None else max(0, start - last_heading_end),
                )
            )
            word_cursor = end
        if sentence_list:
            sentence_map[line_index] = sentence_list

    flat_sentences = [sentence for line in sorted(sentence_map) for sentence in sentence_map[line]]
    candidates: list[_Candidate] = []
    for index, sentence in enumerate(flat_sentences):
        previous = flat_sentences[index - 1].text if index > 0 else ""
        following = flat_sentences[index + 1].text if index + 1 < len(flat_sentences) else ""
        tag, score = _score_sentence(sentence.text, previous, following)
        if not tag:
            continue
        # Keep section headings and the first spoken beat after a heading calm.
        if sentence.after_heading_gap is not None and sentence.after_heading_gap < 12:
            continue
        # Avoid styling the opening hook too aggressively. Let the speaker establish
        # the reference voice before the first expressive cue.
        if sentence.start_word < 35:
            continue
        candidates.append(_Candidate(sentence=sentence, tag=tag, score=score))

    # Pass 2: rank the whole script and select only sparse, high-value moments.
    if total_words < 90:
        max_tags = 0
    else:
        # About one subtle cue per 125 spoken words. A 250-word one-paragraph
        # script can therefore receive two well-separated cues instead of being
        # artificially limited to one by its visual line layout.
        max_tags = min(8, max(1, math.ceil(total_words / 125)))
    min_gap_words = 70
    tag_limits = {
        "happy": max(1, math.ceil(max_tags * 0.45)) if max_tags else 0,
        "narration": max(1, math.ceil(max_tags * 0.50)) if max_tags else 0,
        "surprised": min(2, max_tags),
        "dramatic": min(1, max_tags),
    }
    selected: list[_Candidate] = []
    counts = {tag: 0 for tag in AUTO_ALLOWED_TAGS}

    for candidate in sorted(candidates, key=lambda item: (-item.score, item.sentence.start_word)):
        if len(selected) >= max_tags:
            break
        if counts[candidate.tag] >= tag_limits[candidate.tag]:
            continue
        position = candidate.sentence.start_word
        if any(abs(position - chosen.sentence.start_word) < min_gap_words for chosen in selected):
            continue
        selected.append(candidate)
        counts[candidate.tag] += 1

    selected_lookup = {
        (item.sentence.line_index, item.sentence.sentence_index): item.tag for item in selected
    }

    # Reconstruct a hidden generation script. Original UI text remains untouched.
    output_lines: list[str] = []
    for line_index, raw_line in enumerate(lines):
        if is_heading(raw_line):
            output_lines.append(raw_line)
            continue
        sentence_list = sentence_map.get(line_index)
        if not sentence_list:
            output_lines.append(raw_line)
            continue
        rendered: list[str] = []
        for sentence in sentence_list:
            tag = selected_lookup.get((line_index, sentence.sentence_index))
            rendered.append(f"[{tag}] {sentence.text}" if tag else sentence.text)
        output_lines.append(" ".join(rendered))

    applied = {tag: count for tag, count in counts.items() if count}
    return EmotionAnalysis(
        tagged_text="\n".join(output_lines).strip(),
        total_words=total_words,
        applied_count=sum(applied.values()),
        protected_headings=protected_headings,
        manual_tags=manual_tags,
        by_tag=applied,
    )
