# SoftMeta Chatterbox TTS Server

## v0.7.0 identity-first voice search

- Rebuilds Generate Voice around a stable identity fingerprint before age, emotion, or performance.
- Over-generates up to 12 Qwen3-TTS attempts and returns only the requested distinct candidates.
- Compares each attempt with saved generated voices and every earlier attempt in the same batch.
- Automatically rejects repeated identities instead of showing the same speaker with a new tune.
- Replaces the confusing “100% different” first-candidate label with a truthful baseline label.
- Shows closest speaker similarity, comparison count, identity code, vocal anatomy and spectral traits.
- Separates age behaviour from identity: age changes texture, projection, breath support and thought grouping, not global playback speed.
- Keeps the professional v0.6.0 UI, five predefined voices, queue, waveform, cutter and multi-audio workflow.
- Use `colab/SoftMeta_Chatterbox_TTS_Colab_v0.7.0.ipynb`.

## v0.6.0 professional workspace and voice diversity update

- Keeps generated candidate preview players stable while the queue polls in the background.
- Starts with Audio 1 only; every added Audio 2–5 tab has a removable minus control.
- Adds a clearer professional workspace layout, stronger visual hierarchy and responsive spacing.
- Adds a direct **Chatterbox TTS API** link under API Docs.
- Bundles five user-provided predefined American male WAV references.
- Expands Qwen3-TTS voice identity families, age-conditioned vocal character and sampling variation.
- Uses a stricter similarity threshold and prevents saving candidates marked too similar.
- Automatically advances the variation seed after each successful candidate batch.
- Use `colab/SoftMeta_Chatterbox_TTS_Colab_v0.7.0.ipynb`.


## v0.5.2 model snapshot fix

- Pins the complete official Qwen3-TTS VoiceDesign revision.
- Uses a validated local model snapshot to avoid stale Hugging Face cache files.
- Installs SoX for the isolated Qwen environment.
- Use `colab/SoftMeta_Chatterbox_TTS_Colab_v0.7.0.ipynb`.


## v0.5.1 hotfix

- Fixed the Qwen3-TTS Colab verification cell syntax error.
- No engine, UI workflow, or voice-generation behaviour was changed.
- Use `colab/SoftMeta_Chatterbox_TTS_Colab_v0.7.0.ipynb`.

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
soft-meta/Chatterbox-TTS-Server@v0.7.0
        ↓
Google Colab L4, local Python or Docker
```

This project has no runtime dependency on Devnen repositories.

## Google Colab

Open `colab/SoftMeta_Chatterbox_TTS_Colab_v0.7.0.ipynb`, select an L4 GPU and
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
