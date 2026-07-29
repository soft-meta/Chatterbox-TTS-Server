# SoftMeta Chatterbox TTS Server

## v0.5.1 hotfix

- Fixed the Qwen3-TTS Colab verification cell syntax error.
- No engine, UI workflow, or voice-generation behaviour was changed.
- Use `colab/SoftMeta_Chatterbox_TTS_Colab_v0.5.1.ipynb`.

A professional, self-hosted speech studio maintained by **SoftMeta**. The server,
queue, browser UI, waveform tools and audio editor are SoftMeta code. Chatterbox
inference uses the official MIT-licensed Chatterbox package from Resemble AI
through `soft-meta/chatterbox-v2`.

## v0.5.0 highlights

### Qwen3-TTS Unique Voice Designer

- Replaces Parler-TTS with the official Qwen3-TTS VoiceDesign model
- Generates 2, 3 or 4 new fictional voice candidates in one request
- Explicit speaker age, male/female gender, US English and General American accent
- Age-aware cadence based on phrase length, thought pauses and breathing
- Does **not** globally slow or stretch the generated reference audio
- Varies pitch, resonance, texture, articulation, personality and cadence by seed
- Preview and download every candidate before saving
- Save only the selected candidate as a reusable Chatterbox reference voice
- Optional SpeechBrain ECAPA speaker-difference check against saved generated voices
- Candidate metadata, prompt, seed and optional embedding are saved with the voice

The Voice Designer does not imitate a named real person. Age and identity are
model-generated approximations, so listen before saving.

### Multi-audio studio

- Audio 1 and Audio 2 by default, expandable to Audio 5
- Separate title, script, voice and settings per workspace
- Generate one audio or schedule all prepared workspaces with **Generate All**
- Sequential server-side GPU queue for stable L4 use
- Queue Monitor with live words, percentage, elapsed time, ETA and audio duration
- Preview and download completed audio while the next job is running
- Remove All and removable Audio 3–5 tabs

### Generated audio tools

- Main waveform with live playhead and time display
- Browser audio fallback if waveform drawing fails
- Start and End selection, zoom, fit, wheel scroll and drag-to-pan
- Preview Selected, Download Selected, Part One and Part Two
- Original generated WAV remains unchanged

## Repository relationship

```text
soft-meta/chatterbox-v2@v0.2.1
        ↓
soft-meta/Chatterbox-TTS-Server@v0.5.1
        ↓
Google Colab L4, local Python or Docker
```

This project has no runtime dependency on Devnen repositories.

## Google Colab

Open `colab/SoftMeta_Chatterbox_TTS_Colab_v0.5.1.ipynb`, select an L4 GPU and
run all cells. The notebook creates two isolated environments:

- Chatterbox main server environment
- Qwen3-TTS VoiceDesign environment

The first Generate Voice request downloads the Qwen3-TTS VoiceDesign model.

## Local installation

Python 3.11 and compatible PyTorch/CUDA installations are recommended. Use a
separate Python environment for `requirements-voice.txt` and set
`SOFTMETA_VOICE_PYTHON` to that environment's Python executable.

## Licence

SoftMeta server and UI code are MIT licensed. Chatterbox remains MIT licensed by
Resemble AI. Qwen3-TTS and SpeechBrain are Apache-2.0 licensed. See
`THIRD_PARTY_NOTICES.md`.
