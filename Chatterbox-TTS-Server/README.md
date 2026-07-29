# SoftMeta Chatterbox TTS Server

A professional, self-hosted speech studio maintained by **SoftMeta**. The
server, queue, browser interface, waveform tools and audio editor are SoftMeta
code. Chatterbox inference is provided by the official MIT-licensed Chatterbox
package from Resemble AI through `soft-meta/chatterbox-v2`.

## v0.4.0 highlights

### Age-aware Natural Human Voice Designer

- Explicit **Speaker Age** field from 18 to 110
- Explicit **Male** or **Female** selection
- **US English (American accent)** is enforced for generated voice references
- Main emotion choices for warm, calm, reflective, concerned, serious or hopeful delivery
- Age-specific pacing and vocal behaviour:
  - 50s: slightly slower and thoughtful
  - 60s: slow and clear
  - 70s: noticeably slow and considered
  - 80s: very slow and careful
  - 90+: very slow, fragile but understandable and mentally present
- Automatic Natural Human Voice Formula that discourages narrator, announcer,
  customer-service and synthetic AI rhythms
- Stable Parler-TTS speaker identities selected by gender and seed
- Gentle age-tempo correction for the generated reference sample
- Automatic recommended final Chatterbox speed after a generated voice is selected
- Saved JSON profile beside each generated reference WAV

Text-described age, accent and identity are still approximate. The generated
reference is intended as a practical voice-design starting point, not a promise
of an exact biological age or a specific real person.

### Professional multi-audio studio

- Chatterbox Original English is the default model
- Motivational Speech is the default preset
- Audio 1 and Audio 2 are available by default
- Add workspaces up to Audio 5
- Remove Audio 3, 4 or 5 with the minus button
- Separate title, script, voice and generation settings for every workspace
- Generate one audio or schedule all prepared workspaces with **Generate All**
- Use **Remove All** after the queue finishes to clear completed work

### Voice workflows

- Predefined voices
- Uploaded reference voice cloning
- Voice preview before generation
- Age-aware **Generate Voice** reference creation using Parler-TTS Mini v1.1
- Generated references can be previewed, downloaded, saved and selected for
  Chatterbox long-form cloning

Only use voices and voice descriptions that respect consent and applicable law.

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
- Live playhead, current time and seek control
- Download WAV remains available if waveform loading fails
- Mouse time and click-to-set Start or End
- Zoom, fit, horizontal wheel scrolling and drag-to-pan
- Preview Selected and Download Selected WAV
- Part One and Part Two downloads with title-based filenames
- Original generated WAV remains unchanged by cutting

## Repository relationship

```text
soft-meta/chatterbox-v2@v0.2.1
        ↓
soft-meta/Chatterbox-TTS-Server@v0.4.0
        ↓
Google Colab L4, local Python or Docker
```

This repository has no runtime dependency on `devnen/Chatterbox-TTS-Server` or
`devnen/chatterbox-v2`.

## Google Colab

Open `colab/SoftMeta_Chatterbox_TTS_Colab_v0.4.0.ipynb`, select an L4 GPU and
run every cell from top to bottom. The notebook uses SoftMeta engine `v0.2.1`
and server/UI `v0.4.0`.

Generate Voice runs Parler-TTS in a separate virtual environment because
Parler-TTS 0.2.3 and Chatterbox 0.1.7 require incompatible Transformers
versions. The Colab notebook creates this environment automatically.

## Local installation

Python 3.11 and a compatible PyTorch installation are recommended.

```bash
git clone --branch v0.2.1 https://github.com/soft-meta/chatterbox-v2.git
git clone --branch v0.4.0 https://github.com/soft-meta/Chatterbox-TTS-Server.git

python -m pip install chatterbox-tts==0.1.7
python -m pip install --no-deps -e ./chatterbox-v2
python -m pip install -r ./Chatterbox-TTS-Server/requirements.txt

cd Chatterbox-TTS-Server
python start.py
```

Open `http://127.0.0.1:8004`.

## Reference voice guidance

For Chatterbox cloning, use a clean 10–20 second recording from one speaker
with no music, echo or strong background noise. The reference recording has a
large influence on identity and accent.

## Licence

SoftMeta server and UI code are MIT licensed. Chatterbox software and model
technology remain the work of Resemble AI. Generate Voice uses Parler-TTS under
Apache-2.0. See `THIRD_PARTY_NOTICES.md`.
