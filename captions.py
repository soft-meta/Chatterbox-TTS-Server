from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _stamp(seconds: float, srt: bool) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    sep = "," if srt else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{milli:03d}"


def _groups(words: list[dict[str, Any]], max_words: int = 11, max_seconds: float = 4.5) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        current.append(word)
        duration = float(current[-1]["end"]) - float(current[0]["start"])
        punct = bool(re.search(r"[.!?][\"'”’)]?$", str(word.get("word", "")).strip()))
        if len(current) >= max_words or duration >= max_seconds or (punct and len(current) >= 5):
            groups.append(current); current = []
    if current:
        groups.append(current)
    return groups


def write_caption_files(words: list[dict[str, Any]], srt_path: Path, vtt_path: Path) -> None:
    groups = _groups(words)
    srt_lines: list[str] = []
    vtt_lines: list[str] = ["WEBVTT", ""]
    for index, group in enumerate(groups, start=1):
        start = float(group[0]["start"])
        end = max(start + 0.30, float(group[-1]["end"]))
        text = "".join(str(w.get("word", "")) for w in group).strip()
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        srt_lines.extend([str(index), f"{_stamp(start, True)} --> {_stamp(end, True)}", text, ""])
        vtt_lines.extend([f"{_stamp(start, False)} --> {_stamp(end, False)}", text, ""])
    srt_path.write_text("\n".join(srt_lines).rstrip() + "\n", encoding="utf-8")
    vtt_path.write_text("\n".join(vtt_lines).rstrip() + "\n", encoding="utf-8")


def fallback_words(text: str, start: float, duration: float) -> list[dict[str, Any]]:
    raw = re.findall(r"\S+", text)
    if not raw:
        return []
    step = duration / len(raw)
    return [
        {"start": start + i * step, "end": start + (i + 1) * step, "word": (" " if i else "") + word}
        for i, word in enumerate(raw)
    ]


def aligned_expected_words(expected_text: str, asr_words: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    """Use ASR timing while keeping the creator's expected spoken text in captions."""
    import difflib

    expected_tokens = re.findall(r"\S+", expected_text)
    if not expected_tokens:
        return []
    if not asr_words:
        return fallback_words(expected_text, 0.0, duration)

    def norm(value: str) -> str:
        return "".join(ch.lower() for ch in value if ch.isalnum() or ch == "'")

    ref = [norm(token) for token in expected_tokens]
    hyp = [norm(str(item.get("word", ""))) for item in asr_words]
    matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    times: list[tuple[float, float] | None] = [None] * len(ref)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            j = block.b + offset
            if j < len(asr_words):
                item = asr_words[j]
                times[block.a + offset] = (float(item["start"]), float(item["end"]))

    # Interpolate unmatched expected words between nearby matched anchors.
    matched = [i for i, value in enumerate(times) if value is not None]
    if not matched:
        return fallback_words(expected_text, 0.0, duration)
    for i in range(len(times)):
        if times[i] is not None:
            continue
        left = max((j for j in matched if j < i), default=None)
        right = min((j for j in matched if j > i), default=None)
        if left is None:
            right_start = times[right][0] if right is not None else duration
            step = max(0.08, right_start / max(right or 1, 1))
            start = max(0.0, right_start - step * ((right or 0) - i))
            times[i] = (start, min(right_start, start + step * 0.9))
        elif right is None:
            left_end = times[left][1]
            remaining = len(times) - left - 1
            step = max(0.08, (max(duration, left_end + 0.1) - left_end) / max(remaining, 1))
            start = left_end + step * (i - left - 1)
            times[i] = (start, min(duration, start + step * 0.9))
        else:
            left_end = times[left][1]
            right_start = times[right][0]
            count = right - left - 1
            step = max(0.05, (right_start - left_end) / max(count + 1, 1))
            start = left_end + step * (i - left)
            times[i] = (start, min(right_start, start + step * 0.85))

    result: list[dict[str, Any]] = []
    for i, token in enumerate(expected_tokens):
        start, end = times[i] or (0.0, duration)
        result.append({"start": max(0.0, start), "end": max(start + 0.03, end), "word": (" " if i else "") + token})
    return result
