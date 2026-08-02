# SoftMeta Chatterbox TTS Server v0.8.0

This release replaces Qwen3-TTS VoiceDesign with MOSS VoiceGenerator for more
varied fictional speaker identities and more natural age behaviour.

## MOSS unique voice generation

- Uses `OpenMOSS-Team/MOSS-VoiceGenerator` as the primary Generate Voice model
- Builds a distinct vocal identity before applying age, emotion and cadence
- Keeps older voices diverse instead of forcing every senior into one low narrator tone
- Uses General American speech with varied social and conversational backgrounds
- Avoids global slowdown, stretched vowels and fixed pause intervals
- Uses official MOSS audio sampling controls with controlled per-attempt variation

## Candidate selection

- Searches up to 12 internal attempts for the requested 1–4 candidates
- Compares each attempt with saved voices and earlier attempts in the same batch
- Rejects candidates above the speaker-similarity threshold
- Adds acoustic screening for excessive dead air, clipping, low level and weak dynamics
- Shows Naturalness score, pause ratio, level and dynamic range in the UI
- Reports duplicate-identity and low-quality rejection counts separately

## Runtime

- Colab uses Python 3.11 for Chatterbox and Python 3.12 for MOSS
- Installs the official `OpenMOSS/MOSS-TTS` runtime in an isolated environment
- Pins MOSS VoiceGenerator snapshot revision `97521ec`
- Keeps `soft-meta/chatterbox-v2@v0.2.1` unchanged

## Compatibility

All v0.7.0 studio features remain available, including stable candidate players,
Audio 1 as the only default tab, removable extra tabs, five predefined voices,
queue monitoring, waveform editing and the FastAPI documentation link.
