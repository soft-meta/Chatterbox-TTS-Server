# SoftMeta Chatterbox TTS Server

A professional self-hosted speech studio maintained by **SoftMeta**. The server,
queue, browser UI, waveform tools and audio editor are SoftMeta code. Long-form
speech generation uses the official Chatterbox package through
`soft-meta/chatterbox-v2`.

## v0.8.0: MOSS unique voice generation

The **Generate Voice** workflow now uses the official
`OpenMOSS-Team/MOSS-VoiceGenerator` model instead of Qwen3-TTS.

- Creates fictional American speaker timbres directly from text instructions
- Builds speaker identity before age, emotion and delivery style
- Keeps older voices diverse: bass, baritone, tenor, alto, contralto and lighter voices
- Does not globally slow audio or stretch words to imitate age
- Uses irregular thought and breathing pauses rather than fixed stop-start pacing
- Over-generates up to 12 attempts and returns only the requested 1–4 candidates
- Compares every attempt with saved voices and same-batch attempts using SpeechBrain ECAPA
- Rejects repeated identities above the configured similarity threshold
- Rejects broken samples with excessive dead air, clipping or unusably low speech level
- Shows a Naturalness score with pause, level and dynamic-range diagnostics
- Saves the selected candidate as a reusable Chatterbox reference voice

The model does not imitate a named real person. Age and identity are generated
approximations, so preview each candidate before saving it.

## Studio features

- Audio 1 is the only default workspace; add Audio 2–5 when needed
- Removable minus controls for added audio workspaces
- Sequential GPU queue with Generate All and Queue Monitor
- Stable candidate preview players
- Five bundled predefined American male voices
- Import additional predefined or cloning WAV files
- Main waveform, live playhead, zoom, pan and audio cutter
- Download full audio, selected audio, Part One or Part Two
- FastAPI documentation at `/docs`
- Professional responsive light and dark interface

## Repository relationship

```text
soft-meta/chatterbox-v2@v0.2.1
        ↓
soft-meta/Chatterbox-TTS-Server@v0.8.0
        ↓
Google Colab L4, local Python or Docker
```

This project has no runtime dependency on Devnen repositories.

## Google Colab

Open `colab/SoftMeta_Chatterbox_TTS_Colab_v0.8.0.ipynb`, select an L4 GPU and
run all cells. The notebook creates two isolated environments:

- Python 3.11 Chatterbox server environment
- Python 3.12 MOSS VoiceGenerator environment

The first Generate Voice request downloads the pinned MOSS VoiceGenerator model
snapshot. The model download is several gigabytes.

## Local installation

Use Python 3.11 for the main server. Create a separate clean environment for
MOSS VoiceGenerator, preferably Python 3.12, install the official
`OpenMOSS/MOSS-TTS` repository, then install `requirements-voice.txt`.
Set these environment variables before starting the server:

```text
SOFTMETA_VOICE_PYTHON=/path/to/moss/environment/python
SOFTMETA_MOSS_MODEL_DIR=/path/to/moss/model/cache
```

## Voice rights

Use only reference voices you own or have permission to clone. Generated
fictional candidates should not be presented as a real named person.

## Licence

SoftMeta server and UI code are MIT licensed. Chatterbox remains MIT licensed by
Resemble AI. MOSS VoiceGenerator and SpeechBrain are Apache-2.0 licensed. See
`THIRD_PARTY_NOTICES.md`.
