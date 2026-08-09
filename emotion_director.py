from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from utils import count_words
from speech_pipeline import key_emphasis_for_sentence

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
    rf"^\s*(?:#{{1,6}}\s+|(?:number\s+(?:\d+|{_NUMBER_WORDS})|{_ORDINALS}|\d{{1,2}})\s*[.):-])\s*",
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
        ("the good news", 4.2), ("good news", 3.7), ("fortunately", 3.1),
        ("thankfully", 3.1), ("the encouraging part", 3.5), ("there is hope", 3.7),
        ("you can improve", 2.6), ("can improve", 2.2), ("you can change", 2.5),
        ("can change", 2.1), ("things got better", 3.1), ("began to improve", 2.8),
        ("started to improve", 2.8), ("felt better", 2.4), ("feel better", 2.1),
        ("made a real difference", 3.0), ("changed my life for the better", 3.5),
        ("I was grateful", 2.9), ("I am grateful", 2.9), ("happier", 2.1),
        ("you can still", 1.8), ("can still", 1.4), ("simple change", 1.4),
        ("can help", 1.2), ("may help", 1.1), ("help you", 1.0), ("relief", 2.0),
        ("hope", 1.5), ("better", 1.0), ("improve", 1.0), ("improved", 1.0),
        ("safer", 1.2), ("stronger", 1.0), ("easier", 0.9), ("progress", 0.9),
        ("protect yourself", 1.2),
    ),
    "narration": (
        ("I wish I had", 4.1), ("wish I had", 3.6), ("I learned too late", 4.1),
        ("learned too late", 3.8), ("by the time I realized", 3.4),
        ("I regret", 3.8), ("my biggest regret", 4.1), ("unfortunately", 2.6),
        ("cost me years", 3.3), ("cost me", 2.1), ("years I cannot get back", 4.1),
        ("years I can't get back", 4.1), ("lost years", 3.2), ("I ignored", 2.3),
        ("I was lonely", 3.1), ("felt alone", 2.8), ("I suffered", 2.8),
        ("it hurt", 2.2), ("painful lesson", 3.1), ("hard lesson", 2.2),
        ("one of my mistakes", 2.7), ("one of the biggest mistakes", 3.5),
        ("now pay attention", 3.4), ("pay attention", 3.0), ("remember this", 2.6),
        ("keep this in mind", 2.6), ("this is important", 2.8), ("the important thing", 2.7),
        ("what matters", 2.4), ("this is why", 2.3), ("be careful", 2.4),
        ("follow that advice", 2.3), ("mistake", 1.2), ("regret", 1.5),
        ("pain", 0.8), ("painful", 1.1), ("hurt", 0.9), ("lost", 0.7),
        ("lonely", 1.0), ("suffered", 1.0), ("suffering", 1.0), ("wrong", 0.8),
        ("ignored", 0.8), ("illness", 0.8), ("sick", 0.9), ("difficult", 0.7),
        ("warning", 1.2), ("risk", 0.8),
    ),
    "surprised": (
        ("I couldn't believe", 4.1), ("I could not believe", 4.1),
        ("what surprised me", 4.1), ("you may be surprised", 3.7),
        ("you might be surprised", 3.7), ("I was surprised", 3.7),
        ("to my surprise", 3.6), ("I never expected", 3.6),
        ("most people don't realize", 3.1), ("most people do not realize", 3.1),
        ("few people realize", 3.1), ("not always the whole story", 3.7),
        ("that is not always the whole story", 4.0), ("this is not always the whole story", 4.0),
        ("what many people miss", 3.2), ("you may think", 2.2), ("you might think", 2.2),
        ("unexpected", 2.1), ("surprising", 2.4), ("I discovered", 1.9),
        ("I finally realized", 2.1), ("the truth is", 1.6), ("suddenly", 1.0),
        ("I did not expect", 2.4), ("I didn't expect", 2.4),
    ),
    "dramatic": (
        ("call 911", 5.0), ("call emergency services", 5.0), ("medical emergency", 4.8),
        ("seek emergency care", 4.8), ("could save your life", 4.6),
        ("may save your life", 4.4), ("life-threatening", 4.6),
        ("do not ignore", 3.7), ("never ignore", 3.9), ("warning sign", 3.3),
        ("serious warning", 3.7), ("urgent", 2.6),
    ),
}

_THRESHOLDS = {"happy": 2.15, "narration": 2.15, "surprised": 2.55, "dramatic": 4.10}
_LABELS = {"happy": "warm", "narration": "reflective", "surprised": "surprised", "dramatic": "serious"}

# Secondary, calm emphasis patterns used only when a script has too few strong
# lexical candidates. These create narration cues, never comedy or anger.
_ADVICE_FALLBACK_RE = re.compile(
    r"(?:^|\b)(?:now\s+pay\s+attention|remember|keep\s+in\s+mind|be\s+careful|try\s+|"
    r"make\s+sure|notice\s+|avoid\s+|consider\s+|follow\s+|ask\s+your\s+doctor|talk\s+to\s+your\s+doctor|"
    r"the\s+important\s+thing|what\s+matters|this\s+matters|this\s+is\s+why|one\s+mistake|the\s+next\s+mistake|"
    r"mistake\s+can|risk\s+of|warning\s+sign)\b",
    flags=re.IGNORECASE,
)


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
    source: str = "semantic"


@dataclass(slots=True)
class EmotionAnalysis:
    tagged_text: str
    total_words: int
    applied_count: int
    protected_headings: int
    manual_tags: int
    by_tag: dict[str, int]
    placements: list[dict[str, Any]]
    mode: str = "Serious Senior Advisor"
    avatar_window_words: int = 0
    avatar_applied_count: int = 0
    avatar_target_count: int = 0

    def public_summary(self, *, include_text: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "mode": self.mode,
            "total_words": self.total_words,
            "applied_count": self.applied_count,
            "protected_headings": self.protected_headings,
            "manual_tags": self.manual_tags,
            "by_tag": dict(self.by_tag),
            "labels": {_LABELS[tag]: count for tag, count in self.by_tag.items() if count},
            "placements": list(self.placements),
            "key_emphasis_count": sum(1 for item in self.placements if item.get("source") == "key-emphasis"),
            "retention_reset_count": sum(1 for item in self.placements if item.get("source") == "retention-reset"),
            "high_confidence_count": sum(1 for item in self.placements if float(item.get("confidence", 0.0)) >= 0.82),
            "medium_confidence_count": sum(1 for item in self.placements if 0.68 <= float(item.get("confidence", 0.0)) < 0.82),
            "avatar_window_words": self.avatar_window_words,
            "avatar_applied_count": self.avatar_applied_count,
            "avatar_target_count": self.avatar_target_count,
            "avatar_reset_count": sum(1 for item in self.placements if item.get("source") == "avatar-reset"),
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
    if clean.endswith(":") and len(words) <= 12 and clean.count(".") == 0 and clean.count("?") == 0:
        return True
    letters = "".join(ch for ch in clean if ch.isalpha())
    if letters and letters.upper() == letters and len(words) <= 12:
        return True
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
            phrase_lower = phrase.lower()
            if phrase_lower in current:
                scores[tag] += weight
            elif phrase_lower in context_before or phrase_lower in context_after:
                scores[tag] += weight * 0.14

    # Calm discourse cues help an advisor sound human without making every sentence
    # emotional. These only strengthen an already meaningful transition.
    if re.match(r"^(?:but|however|yet)\b", current):
        scores["narration"] += 0.75
        if "not always" in current or "not the whole" in current:
            scores["surprised"] += 1.4
    if re.match(r"^(?:now|remember|importantly|instead)\b", current):
        scores["narration"] += 0.65
    if _ADVICE_FALLBACK_RE.search(current):
        scores["narration"] += 0.95

    first_person = bool(re.search(r"\b(?:i|i'm|i've|my|me|we|our)\b", current))
    if first_person and scores["narration"] > 0:
        scores["narration"] += 0.45
    if "!" in clean:
        for tag in ("happy", "surprised"):
            if scores[tag] > 0:
                scores[tag] += 0.12

    if scores["dramatic"] and not any(
        phrase in current
        for phrase in (
            "call 911", "emergency", "save your life", "life-threatening",
            "do not ignore", "never ignore", "serious warning",
        )
    ):
        scores["dramatic"] *= 0.60

    tag, score = max(scores.items(), key=lambda item: item[1])
    if score < _THRESHOLDS[tag]:
        return None, score
    return tag, score


def _fallback_narration_score(sentence: _Sentence) -> float:
    clean = strip_turbo_tags(sentence.text).strip()
    words = count_words(clean)
    if words < 7 or words > 48 or _CTA_RE.search(clean):
        return 0.0
    if _ADVICE_FALLBACK_RE.search(clean):
        return 2.30
    lower = clean.lower()
    if re.match(r"^(?:but|however|now|remember)\b", lower) and any(
        token in lower for token in ("mistake", "important", "advice", "health", "doctor", "risk", "problem", "better")
    ):
        return 2.20
    return 0.0


def _candidate_confidence(candidate: _Candidate) -> float:
    threshold = _THRESHOLDS.get(candidate.tag, 2.0)
    if candidate.source == "key-emphasis":
        return 0.97 if candidate.score >= 4.5 else 0.90
    if candidate.source == "retention-reset":
        return 0.72
    if candidate.source == "advice-fallback":
        return 0.69
    margin = max(0.0, candidate.score - threshold)
    return float(min(0.98, 0.72 + margin * 0.10))


def analyze_serious_senior_advisor(text: str) -> EmotionAnalysis:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    manual_tags = len(_TAG_RE.findall(normalized))
    # Serious Advisor mode owns expression direction. Remove any existing Turbo tags
    # first, then deterministically rebuild the safe visible tag layer. This makes
    # frontend preview and backend generation idempotent and keeps forbidden tags out.
    sanitized = strip_turbo_tags(normalized)
    lines = sanitized.split("\n")
    total_words = count_words(sanitized)
    protected_headings = 0

    # Pass 1: read the whole script, identify headings, sentence positions and
    # surrounding context. No tags are inserted yet.
    sentence_map: dict[int, list[_Sentence]] = {}
    word_cursor = 0
    last_heading_end: int | None = None
    for line_index, raw_line in enumerate(lines):
        if is_heading(raw_line):
            protected_headings += 1
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
    candidate_keys: set[tuple[int, int]] = set()
    for index, sentence in enumerate(flat_sentences):
        previous = flat_sentences[index - 1].text if index > 0 else ""
        following = flat_sentences[index + 1].text if index + 1 < len(flat_sentences) else ""
        tag, score = _score_sentence(sentence.text, previous, following)
        source = "semantic"
        emphasis_tag, emphasis_score = key_emphasis_for_sentence(sentence.text)
        if emphasis_tag and emphasis_score > score:
            tag, score, source = emphasis_tag, emphasis_score, "key-emphasis"
        if not tag:
            continue
        # Headings stay clean and the first few spoken words after a heading remain
        # neutral so numbered sections do not start with a sudden effect.
        if sentence.after_heading_gap is not None and sentence.after_heading_gap < 10 and source != "key-emphasis":
            continue
        # Let the speaker establish the voice before the first expressive cue.
        if sentence.start_word < 28:
            continue
        key = (sentence.line_index, sentence.sentence_index)
        candidates.append(_Candidate(sentence=sentence, tag=tag, score=score, source=source))
        candidate_keys.add(key)

    # Pass 2: add calm advisory fallback candidates only when the semantic pass would
    # otherwise under-direct a medium/long script. This is still based on meaning and
    # sentence role, not arbitrary every-Nth-sentence tagging.
    for sentence in flat_sentences:
        key = (sentence.line_index, sentence.sentence_index)
        if key in candidate_keys or sentence.start_word < 28:
            continue
        if sentence.after_heading_gap is not None and sentence.after_heading_gap < 10:
            continue
        score = _fallback_narration_score(sentence)
        if score > 0:
            candidates.append(_Candidate(sentence=sentence, tag="narration", score=score, source="advice-fallback"))

    if total_words < 80:
        max_tags = 0
        desired_min = 0
    else:
        # Professional but perceptible direction. Long 8–12 minute scripts need
        # enough local expression beats to be audible without turning into acting.
        # Aim for roughly one potential cue per 120 words, capped at fourteen; medium
        # scripts still receive at least two meaningful cues when content supports it.
        max_tags = min(14, max(1, math.ceil(total_words / 120)))
        if total_words >= 220:
            desired_min = min(max_tags, max(2, total_words // 170))
        else:
            desired_min = 1

    min_gap_words = 44
    tag_limits = {
        "happy": max(1, math.ceil(max_tags * 0.40)) if max_tags else 0,
        "narration": max(1, math.ceil(max_tags * 0.62)) if max_tags else 0,
        "surprised": min(3, max_tags),
        "dramatic": min(2 if total_words >= 900 else 1, max_tags),
    }
    selected: list[_Candidate] = []
    counts = {tag: 0 for tag in AUTO_ALLOWED_TAGS}
    # Reserve a small part of the long-form cue budget for neutral retention resets.
    # This prevents ordinary advice-fallback candidates from consuming all eight slots
    # before we have a chance to break up a 90+ second flat stretch.
    reserved_reset_slots = 0
    if total_words >= 500 and max_tags >= 3:
        reserved_reset_slots = 1 if max_tags < 7 else 2 if max_tags < 12 else 3
    semantic_cap = max(0, max_tags - reserved_reset_slots)

    def can_select(candidate: _Candidate) -> bool:
        if _candidate_confidence(candidate) < 0.68:
            return False
        if counts[candidate.tag] >= tag_limits[candidate.tag]:
            return False
        position = candidate.sentence.start_word
        if any(abs(position - chosen.sentence.start_word) < min_gap_words for chosen in selected):
            return False
        # Avoid repeating the same emotional color too close together even when the
        # raw word-gap rule would allow it.
        if any(
            candidate.tag == chosen.tag
            and abs(position - chosen.sentence.start_word) < 120
            for chosen in selected
        ):
            return False
        return True

    for candidate in sorted(candidates, key=lambda item: (-item.score, item.sentence.start_word)):
        if len(selected) >= semantic_cap:
            break
        if not can_select(candidate):
            continue
        selected.append(candidate)
        counts[candidate.tag] += 1

    # If a medium/long advice script has enough suitable candidates but score ranking
    # plus diversity rules left it below the intended minimum, make one extra pass
    # using the calm narration fallbacks. Never fabricate happy/angry/comedic emotion.
    if len(selected) < desired_min:
        for candidate in sorted(
            (item for item in candidates if item.source == "advice-fallback"),
            key=lambda item: item.sentence.start_word,
        ):
            if len(selected) >= min(desired_min, semantic_cap) or len(selected) >= semantic_cap:
                break
            if candidate in selected or not can_select(candidate):
                continue
            selected.append(candidate)
            counts[candidate.tag] += 1

    # Long-form retention reset: if the semantic cues leave a flat stretch longer
    # than roughly 80-90 seconds at senior-friendly pace, add one neutral narration
    # cue near the middle. This never adds comedy, anger or fabricated emotion.
    if total_words >= 500 and max_tags and len(selected) < max_tags:
        selected.sort(key=lambda item: item.sentence.start_word)
        attempted_targets: set[int] = set()
        while len(selected) < max_tags:
            positions = [40] + [item.sentence.start_word for item in selected] + [total_words]
            gaps = [(b - a, a, b) for a, b in zip(positions, positions[1:])]
            gap, left, right = max(gaps, key=lambda item: item[0])
            if gap <= 175:
                break
            target = (left + right) // 2
            if target in attempted_targets:
                break
            attempted_targets.add(target)
            eligible = []
            for sentence in flat_sentences:
                if sentence.start_word < 45 or sentence.start_word >= total_words - 25:
                    continue
                if sentence.after_heading_gap is not None and sentence.after_heading_gap < 10:
                    continue
                clean = strip_turbo_tags(sentence.text).strip()
                if _CTA_RE.search(clean) or not (7 <= count_words(clean) <= 45):
                    continue
                if any(abs(sentence.start_word - item.sentence.start_word) < 62 for item in selected):
                    continue
                eligible.append(sentence)
            if not eligible or counts["narration"] >= tag_limits["narration"]:
                break
            chosen_sentence = min(eligible, key=lambda item: abs(item.start_word - target))
            reset = _Candidate(
                sentence=chosen_sentence,
                tag="narration",
                score=2.05,
                source="retention-reset",
            )
            selected.append(reset)
            counts["narration"] += 1
            selected.sort(key=lambda item: item.sentence.start_word)

    selected.sort(key=lambda item: item.sentence.start_word)
    selected_lookup = {
        (item.sentence.line_index, item.sentence.sentence_index): item for item in selected
    }

    # Reconstruct the same canonical tagged script returned to the browser and used
    # again by the backend. Headings and blank-line structure stay untouched.
    output_lines: list[str] = []
    placements: list[dict[str, Any]] = []
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
            chosen = selected_lookup.get((line_index, sentence.sentence_index))
            if chosen:
                tag = chosen.tag
                rendered.append(f"[{tag}] {sentence.text}")
                placements.append(
                    {
                        "tag": tag,
                        "label": _LABELS[tag],
                        "source": chosen.source,
                        "confidence": round(_candidate_confidence(chosen), 2),
                        "word_position": sentence.start_word,
                        "line": line_index + 1,
                        "excerpt": sentence.text[:120],
                    }
                )
            else:
                rendered.append(sentence.text)
        output_lines.append(" ".join(rendered))

    applied = {tag: count for tag, count in counts.items() if count}
    return EmotionAnalysis(
        tagged_text="\n".join(output_lines).strip(),
        total_words=total_words,
        applied_count=sum(applied.values()),
        protected_headings=protected_headings,
        manual_tags=manual_tags,
        by_tag=applied,
        placements=placements,
    )



def analyze_turbo_avatar_performance(
    text: str,
    *,
    target_wpm: int = 148,
    avatar_minutes: float = 5.0,
) -> EmotionAnalysis:
    """Front-load Turbo-native expression for the on-camera avatar window.

    Original Chatterbox continues to use ``analyze_serious_senior_advisor`` unchanged.
    Turbo starts from that conservative semantic plan, then adds only supported native
    emotion tokens inside the first five-minute avatar zone when long flat stretches
    remain. Semantic candidates are preferred; calm ``[narration]`` resets fill only
    genuine spacing gaps. The tail stays deliberately calmer for B-roll narration.
    """
    base = analyze_serious_senior_advisor(text)
    if base.total_words < 80:
        base.mode = "Turbo Avatar Performance"
        return base

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    sanitized = strip_turbo_tags(normalized)
    lines = sanitized.split("\n")
    total_words = count_words(sanitized)
    avatar_window_words = min(total_words, max(0, int(round(max(target_wpm, 1) * max(avatar_minutes, 0.0)))))
    if avatar_window_words <= 0:
        return base

    # Roughly one meaningful beat every 30-38 seconds at senior-advisor pace,
    # capped at ten in the first five minutes. This is intentionally denser than the
    # default Serious Senior Advisor plan but remains far below sentence-by-sentence acting.
    # A full five-minute avatar window targets ten restrained beats, while shorter
    # scripts scale down naturally. Using ceil here prevents a nominal 5-minute zone
    # (about 740 words at 148 WPM) from stopping at nine cues and leaving a 60-70s
    # flat pocket despite otherwise good semantic placements.
    avatar_target = min(10, max(2, math.ceil(avatar_window_words / 78.0)))
    total_cap = min(14, avatar_target + max(1, math.ceil(max(total_words - avatar_window_words, 0) / 240.0)))

    sentence_map: dict[int, list[_Sentence]] = {}
    flat_sentences: list[_Sentence] = []
    word_cursor = 0
    last_heading_end: int | None = None
    for line_index, raw_line in enumerate(lines):
        if is_heading(raw_line):
            word_cursor += count_words(strip_turbo_tags(raw_line))
            last_heading_end = word_cursor
            continue
        sentence_list: list[_Sentence] = []
        for sentence_index, part in enumerate(_sentence_parts(raw_line)):
            speech_words = count_words(strip_turbo_tags(part))
            start = word_cursor
            end = word_cursor + speech_words
            sentence = _Sentence(
                line_index=line_index,
                sentence_index=sentence_index,
                text=part,
                start_word=start,
                end_word=end,
                after_heading_gap=None if last_heading_end is None else max(0, start - last_heading_end),
            )
            sentence_list.append(sentence)
            flat_sentences.append(sentence)
            word_cursor = end
        if sentence_list:
            sentence_map[line_index] = sentence_list

    # Rebuild selected items from the conservative plan so its semantic decisions and
    # calm tail remain intact. Then enrich only the first-five-minute Turbo zone.
    selected: list[_Candidate] = []
    by_key: dict[tuple[int, int], _Candidate] = {}
    placement_positions = {(int(item.get("line", 0)) - 1, int(item.get("word_position", -1))): item for item in base.placements}
    for sentence in flat_sentences:
        item = placement_positions.get((sentence.line_index, sentence.start_word))
        if not item:
            continue
        candidate = _Candidate(
            sentence=sentence,
            tag=str(item.get("tag") or "narration"),
            score=max(2.0, float(item.get("confidence", 0.75)) * 4.0),
            source=str(item.get("source") or "semantic"),
        )
        selected.append(candidate)
        by_key[(sentence.line_index, sentence.sentence_index)] = candidate

    # Reserve the requested first-five-minute capacity before enrichment. The default
    # planner may spend most of its budget in the calmer tail; Turbo Avatar Performance
    # intentionally reverses that priority for the user's on-camera opening.
    tail_cap = max(0, total_cap - avatar_target)
    front_seed = [item for item in selected if item.sentence.start_word < avatar_window_words]
    tail_seed = [item for item in selected if item.sentence.start_word >= avatar_window_words]
    if len(tail_seed) > tail_cap:
        tail_seed = sorted(tail_seed, key=lambda item: (-item.score, item.sentence.start_word))[:tail_cap]
        selected = front_seed + tail_seed
        by_key = {(item.sentence.line_index, item.sentence.sentence_index): item for item in selected}

    def front_items() -> list[_Candidate]:
        return [item for item in selected if item.sentence.start_word < avatar_window_words]

    def gap_ok(sentence: _Sentence, tag: str, *, min_gap: int = 48) -> bool:
        pos = sentence.start_word
        if pos < 14 or pos >= avatar_window_words:
            return False
        if sentence.after_heading_gap is not None and sentence.after_heading_gap < 8:
            return False
        if any(abs(pos - item.sentence.start_word) < min_gap for item in front_items()):
            return False
        if any(tag == item.tag and abs(pos - item.sentence.start_word) < 74 for item in front_items()):
            return False
        return True

    # Prefer unused high-confidence semantic/key-emphasis moments before adding neutral
    # resets. Earlier positions get a small boost because the avatar section is the
    # retention-critical part of the user's workflow.
    extras: list[_Candidate] = []
    for index, sentence in enumerate(flat_sentences):
        if sentence.start_word >= avatar_window_words or sentence.start_word < 14:
            continue
        if (sentence.line_index, sentence.sentence_index) in by_key:
            continue
        previous = flat_sentences[index - 1].text if index > 0 else ""
        following = flat_sentences[index + 1].text if index + 1 < len(flat_sentences) else ""
        tag, score = _score_sentence(sentence.text, previous, following)
        source = "semantic"
        emphasis_tag, emphasis_score = key_emphasis_for_sentence(sentence.text)
        if emphasis_tag and emphasis_score > score:
            tag, score, source = emphasis_tag, emphasis_score, "key-emphasis"
        if not tag:
            fallback_score = _fallback_narration_score(sentence)
            if fallback_score > 0:
                tag, score, source = "narration", fallback_score, "advice-fallback"
        if not tag:
            continue
        candidate = _Candidate(sentence=sentence, tag=tag, score=score, source=source)
        if _candidate_confidence(candidate) < 0.68:
            continue
        early_bonus = max(0.0, 0.35 * (1.0 - sentence.start_word / max(avatar_window_words, 1)))
        candidate.score += early_bonus
        extras.append(candidate)

    for candidate in sorted(extras, key=lambda item: (-item.score, item.sentence.start_word)):
        if len(front_items()) >= avatar_target or len(selected) >= total_cap:
            break
        if not gap_ok(candidate.sentence, candidate.tag):
            continue
        selected.append(candidate)
        by_key[(candidate.sentence.line_index, candidate.sentence.sentence_index)] = candidate

    # Guarantee the first 30 seconds are not an entirely flat establishing read when a
    # safe sentence exists. Use narration rather than inventing a positive/surprised mood.
    intro_limit = min(82, avatar_window_words)
    if not any(item.sentence.start_word < intro_limit for item in front_items()) and len(selected) < total_cap:
        eligible = [
            sentence for sentence in flat_sentences
            if 14 <= sentence.start_word < intro_limit
            and 7 <= count_words(strip_turbo_tags(sentence.text)) <= 42
            and not _CTA_RE.search(sentence.text)
            and (sentence.after_heading_gap is None or sentence.after_heading_gap >= 8)
        ]
        if eligible:
            chosen = min(eligible, key=lambda sentence: abs(sentence.start_word - 42))
            reset = _Candidate(chosen, "narration", 2.25, "avatar-reset")
            if gap_ok(chosen, "narration", min_gap=34):
                selected.append(reset)
                by_key[(chosen.line_index, chosen.sentence_index)] = reset

    # Fill only genuine first-five-minute flat stretches. This keeps the average spacing
    # around 30-40 seconds without fabricating happy/surprised emotion where the words
    # do not support it.
    attempted: set[int] = set()
    while len(front_items()) < avatar_target and len(selected) < total_cap:
        front = sorted(front_items(), key=lambda item: item.sentence.start_word)
        anchors = [12] + [item.sentence.start_word for item in front] + [avatar_window_words]
        gaps = [(right - left, left, right) for left, right in zip(anchors, anchors[1:])]
        gap, left, right = max(gaps, key=lambda item: item[0])
        if gap <= 86:
            break
        target = (left + right) // 2
        if target in attempted:
            break
        attempted.add(target)
        eligible = []
        for sentence in flat_sentences:
            if not (14 <= sentence.start_word < avatar_window_words):
                continue
            if (sentence.line_index, sentence.sentence_index) in by_key:
                continue
            clean = strip_turbo_tags(sentence.text).strip()
            if _CTA_RE.search(clean) or not (7 <= count_words(clean) <= 42):
                continue
            if sentence.after_heading_gap is not None and sentence.after_heading_gap < 8:
                continue
            if not gap_ok(sentence, "narration", min_gap=46):
                continue
            eligible.append(sentence)
        if not eligible:
            break
        chosen = min(eligible, key=lambda sentence: abs(sentence.start_word - target))
        reset = _Candidate(chosen, "narration", 2.20, "avatar-reset")
        selected.append(reset)
        by_key[(chosen.line_index, chosen.sentence_index)] = reset

    # If the enriched front pushed the plan over the global cap, keep every avatar beat
    # and retain only the earliest/strongest calm-tail items. Original is unaffected.
    if len(selected) > total_cap:
        front = [item for item in selected if item.sentence.start_word < avatar_window_words]
        tail = [item for item in selected if item.sentence.start_word >= avatar_window_words]
        tail = sorted(tail, key=lambda item: (-item.score, item.sentence.start_word))[: max(0, total_cap - len(front))]
        selected = front + tail

    selected.sort(key=lambda item: item.sentence.start_word)
    selected_lookup = {(item.sentence.line_index, item.sentence.sentence_index): item for item in selected}
    counts = {tag: 0 for tag in AUTO_ALLOWED_TAGS}
    for item in selected:
        counts[item.tag] += 1

    output_lines: list[str] = []
    placements: list[dict[str, Any]] = []
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
            chosen = selected_lookup.get((line_index, sentence.sentence_index))
            if chosen:
                rendered.append(f"[{chosen.tag}] {sentence.text}")
                placements.append({
                    "tag": chosen.tag,
                    "label": _LABELS[chosen.tag],
                    "source": chosen.source,
                    "confidence": round(_candidate_confidence(chosen), 2),
                    "word_position": sentence.start_word,
                    "line": line_index + 1,
                    "excerpt": sentence.text[:120],
                    "avatar_zone": sentence.start_word < avatar_window_words,
                })
            else:
                rendered.append(sentence.text)
        output_lines.append(" ".join(rendered))

    applied = {tag: count for tag, count in counts.items() if count}
    avatar_applied = sum(1 for item in placements if item.get("avatar_zone"))
    return EmotionAnalysis(
        tagged_text="\n".join(output_lines).strip(),
        total_words=total_words,
        applied_count=sum(applied.values()),
        protected_headings=base.protected_headings,
        manual_tags=base.manual_tags,
        by_tag=applied,
        placements=placements,
        mode="Turbo Avatar Performance",
        avatar_window_words=avatar_window_words,
        avatar_applied_count=avatar_applied,
        avatar_target_count=avatar_target,
    )
