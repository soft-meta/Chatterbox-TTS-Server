from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping

from config import ROOT

_TAG = re.compile(r"\[(?:happy|surprised|dramatic|narration)\]", re.IGNORECASE)

# High-value terms for senior health/advice narration. The goal is stable spoken
# English, not a giant medical dictionary that may rewrite ordinary words.
DEFAULT_PRONUNCIATIONS: dict[str, str] = {
    "A1C": "A one C",
    "B12": "B twelve",
    "B6": "B six",
    "LDL": "L D L",
    "HDL": "H D L",
    "BMI": "B M I",
    "COPD": "C O P D",
    "CPR": "C P R",
    "CDC": "C D C",
    "FDA": "F D A",
    "MRI": "M R I",
    "CT": "C T",
    "ECG": "E C G",
    "EKG": "E K G",
    "UTI": "U T I",
    "PSA": "P S A",
    "GERD": "G E R D",
    "DVT": "D V T",
    "AFib": "A fib",
    "AFIB": "A fib",
    "COVID-19": "COVID nineteen",
    "911": "nine one one",
}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_TEENS = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"}
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def number_to_words(value: int) -> str:
    if value < 0:
        return "minus " + number_to_words(-value)
    if value < 10:
        return _ONES[value]
    if value < 20:
        return _TEENS[value]
    if value < 100:
        tens, rest = divmod(value, 10)
        return _TENS[tens * 10] + (" " + _ONES[rest] if rest else "")
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        return _ONES[hundreds] + " hundred" + (" " + number_to_words(rest) if rest else "")
    if value < 10000:
        thousands, rest = divmod(value, 1000)
        return number_to_words(thousands) + " thousand" + (" " + number_to_words(rest) if rest else "")
    return str(value)


def _load_custom_map() -> dict[str, str]:
    path = ROOT / "pronunciations.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k): str(v) for k, v in data.items() if str(k).strip() and str(v).strip()} if isinstance(data, dict) else {}


def _protect_tags(text: str) -> tuple[str, list[str]]:
    tags: list[str] = []
    def repl(match: re.Match[str]) -> str:
        tags.append(match.group(0).lower())
        return f"__SMEMOTION{len(tags)-1}__"
    return _TAG.sub(repl, text), tags


def _restore_tags(text: str, tags: list[str]) -> str:
    for i, tag in enumerate(tags):
        text = text.replace(f"__SMEMOTION{i}__", tag)
    return text


def prepare_pronunciation_text(text: str, custom: Mapping[str, str] | None = None) -> str:
    """Normalize high-risk numbers, units and medical abbreviations for English TTS.

    The visible user script is never changed. This is a hidden generation copy used
    by both Original and Turbo. Emotion control tags are preserved for the model-
    specific direction layer.
    """
    working, tags = _protect_tags(text)
    replacements = dict(DEFAULT_PRONUNCIATIONS)
    replacements.update(_load_custom_map())
    if custom:
        replacements.update({str(k): str(v) for k, v in custom.items()})

    # Longest keys first so B12 is handled before a shorter overlapping token.
    for source in sorted(replacements, key=len, reverse=True):
        target = replacements[source]
        working = re.sub(rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])", target, working, flags=re.IGNORECASE)

    # Blood pressure-style readings.
    working = re.sub(
        r"\b(\d{2,3})\s*/\s*(\d{2,3})\b",
        lambda m: f"{number_to_words(int(m.group(1)))} over {number_to_words(int(m.group(2)))}",
        working,
    )
    # Age notation such as 70+.
    working = re.sub(r"\b(\d{1,3})\s*\+", lambda m: f"{number_to_words(int(m.group(1)))} and older", working)
    # Percentages and temperature.
    working = re.sub(r"\b(\d+(?:\.\d+)?)\s*%", lambda m: f"{_decimal_words(m.group(1))} percent", working)
    working = re.sub(r"\b(\d+(?:\.\d+)?)\s*°?\s*F\b", lambda m: f"{_decimal_words(m.group(1))} degrees Fahrenheit", working, flags=re.IGNORECASE)
    working = re.sub(r"\b(\d+(?:\.\d+)?)\s*°?\s*C\b", lambda m: f"{_decimal_words(m.group(1))} degrees Celsius", working, flags=re.IGNORECASE)

    units = [
        (r"mg", "milligrams"), (r"mcg|µg", "micrograms"), (r"g", "grams"),
        (r"mL", "milliliters"), (r"L", "liters"), (r"mmHg", "millimeters of mercury"),
        (r"bpm", "beats per minute"), (r"mm", "millimeters"), (r"cm", "centimeters"),
    ]
    for pattern, spoken in units:
        working = re.sub(
            rf"\b(\d+(?:\.\d+)?)\s*({pattern})\b",
            lambda m, spoken=spoken: f"{_decimal_words(m.group(1))} {spoken}",
            working,
            flags=re.IGNORECASE,
        )

    # Dollar amounts and common numeric ranges.
    working = re.sub(r"\$(\d+(?:\.\d+)?)", lambda m: f"{_decimal_words(m.group(1))} dollars", working)
    working = re.sub(
        r"\b(\d{1,3})\s*[-–]\s*(\d{1,3})\b",
        lambda m: f"{number_to_words(int(m.group(1)))} to {number_to_words(int(m.group(2)))}",
        working,
    )
    # Decimals are safer spoken explicitly; leave ordinary integer years/counts alone.
    working = re.sub(r"\b\d+\.\d+\b", lambda m: _decimal_words(m.group(0)), working)
    working = re.sub(r"[ \t]+", " ", working)
    return _restore_tags(working.strip(), tags)


def _decimal_words(value: str) -> str:
    if "." not in value:
        return number_to_words(int(value))
    whole, fraction = value.split(".", 1)
    return number_to_words(int(whole)) + " point " + " ".join(_ONES[int(ch)] for ch in fraction if ch.isdigit())
