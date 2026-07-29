# SoftMeta Chatterbox TTS Server

A professional, self-hosted speech studio maintained by **SoftMeta**. The
server, queue, browser interface, waveform tools and audio editor are SoftMeta
code. Chatterbox model inference is provided by the official MIT-licensed
Chatterbox package from Resemble AI through `soft-meta/chatterbox-v2`.

## v0.3.0 highlights

### Professional multi-audio studio

- Chatterbox Original English is the default model
- Motivational Speech is the default preset
- Audio 1 and Audio 2 are available by default
- Add workspaces up to Audio 5
- Remove an accidentally added Audio 3, 4 or 5 with the minus button
- Separate title, script, voice and generation settings for every workspace
- Generate one audio or schedule all prepared workspaces with **Generate All**
- Use **Remove All** after the queue finishes to delete completed output files,
  clear titles and scripts, and return to fresh Audio 1 and Audio 2 workspaces

### Voice workflows

- Predefined voices
- Uploaded reference voice cloning
- Voice preview before generation
- **Generate Voice** creates a short text-described reference WAV using
  Parler-TTS Mini v1.1
- Generated voice references can be previewed, downloaded, saved and selected
  for Chatterbox long-form cloning

Text-described voices are approximate. A description can guide broad traits
such as pitch, pace, gender presentation, accent style and recording quality,
but it cannot guarantee an exact age, identity or accent. Only use voices and
voice descriptions that respect consent and applicable law.

### Queue and progress

- Server-side sequential GPU queue for stable L4 operation
- Generate up to five prepared audio jobs without waiting at the browser
- Queue Monitor with selectable titles
- Live generated and remaining percentages
- Live word progress, elapsed time, ETA and estimated final duration
- Minimise the monitor to a bottom-right live status widget
- Preview and download completed audio while the next job is running

### Generated audio and editing

- One custom playback control directly below the waveform
- Live playhead, current time and seek control on the main waveform
- No duplicate browser music player below the graph
- Download WAV remains available if waveform loading fails
- Mouse time and click-to-set Start or End
- Zoom, fit, horizontal wheel scrolling and drag-to-pan
- Custom Start and End fields
- Preview Selected and Download Selected WAV
- Part One and Part Two downloads with title-based filenames
- Original generated WAV remains unchanged by cutting

### Interface polish

- Larger, more readable typography
- Professional light and dark themes
- SoftMeta footer link to `soft-meta/chatterbox-v2`
- No Server Configuration block in the studio
- No fixed “Keep first” quick-time buttons in the cutter

## Repository relationship

```text
soft-meta/chatterbox-v2@v0.2.1
        ↓
soft-meta/Chatterbox-TTS-Server@v0.3.0
        ↓
Google Colab L4, local Python or Docker
```

This repository has no runtime dependency on `devnen/Chatterbox-TTS-Server` or
`devnen/chatterbox-v2`.

## Google Colab

Open `colab/SoftMeta_Chatterbox_TTS_Colab_v0.3.0.ipynb`, select an L4 GPU and
run every cell from top to bottom. The notebook uses the stable SoftMeta engine
`v0.2.1` and server/UI `v0.3.0` releases.

The optional Generate Voice feature downloads a separate Parler-TTS model the
first time it is used. To reduce GPU pressure, SoftMeta unloads Chatterbox,
generates the reference sample, clears the voice-design model, and reloads the
previous Chatterbox model.

## Local installation

Python 3.11 and a compatible PyTorch installation are recommended.

```bash
git clone --branch v0.2.1 https://github.com/soft-meta/chatterbox-v2.git
git clone --branch v0.3.0 https://github.com/soft-meta/Chatterbox-TTS-Server.git

python -m pip install chatterbox-tts==0.1.7
python -m pip install --no-deps -e ./chatterbox-v2
python -m pip install -r ./Chatterbox-TTS-Server/requirements.txt
# Optional Generate Voice feature:
python -m pip install --no-deps -r ./Chatterbox-TTS-Server/requirements-voice.txt

cd Chatterbox-TTS-Server
python start.py
```

Open `http://127.0.0.1:8004`.

## Reference voice guidance

For Chatterbox cloning, use a clean 10–20 second recording from one speaker
with no music, echo or strong background noise. The accent and identity of the
output are strongly influenced by the reference recording. A generation preset
cannot reliably replace the reference speaker’s accent.

## Licence

SoftMeta server and UI code are MIT licensed. Chatterbox software and model
technology remain the work of Resemble AI. The optional Generate Voice feature
uses Parler-TTS under Apache-2.0. See `THIRD_PARTY_NOTICES.md`.
