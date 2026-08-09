# SoftMeta Chatterbox TTS Server

Self-hosted long-form Chatterbox narration studio maintained by **SoftMeta**.

## v1.5.0 Production Quality

**Motivational Speech now uses the same professional production pipeline with both Chatterbox Turbo and Chatterbox Original.** Turbo remains the default for faster English generation; Original keeps its native CFG and Exaggeration controls. Turbo emotion tags are never sent literally to Original: the server maps the same direction to Original-supported settings and strips the control tokens before inference.

### Shared professional pipeline

- **Senior Clear Speech** with conservative American-English text preparation
- **Pronunciation Engine** for high-risk numbers, units and common health terms, plus editable `pronunciations.json`
- **Serious Auto Emotion** with visible tags, heading protection and confidence filtering
- **Intelligent Micro-Pause** and excessive-silence cleanup
- **Age-aware target-band pacing** for 60s, 70s, 80s and 90+ listeners; in-band speech is left untouched
- **Section openings, 30-second intro profile and sparse retention resets**
- **Voice-aware mastering** with adaptive presence, de-essing and gentle compression
- **Reference Voice Quality Analyzer** before professional clone generation
- **Production Quality Gate** after each accepted chunk: acoustic checks, ASR/script comparison and optional reference-speaker consistency
- **Focused Smart Retry** only for a failed chunk; a chunk still failing after two retries is rejected instead of silently shipped
- **48 kHz stereo Video Master** companion WAV for video editing/upload
- **SRT and VTT captions** aligned to the accepted audio while preserving creator-facing script terms
- Optional local **performance feedback** storage for platform, 30-second retention and average view duration comparisons

### Quality verification

The production gate uses Faster-Whisper for English transcript verification and SpeechBrain ECAPA-TDNN for reference-speaker consistency when a clone/reference voice is used. Both models load lazily. If a verifier cannot load, acoustic checks remain active rather than preventing the server from starting.

The first professional generation in a fresh Colab runtime can take extra time while the verification models download. Subsequent generations reuse the local Hugging Face/model cache.

### Output

Each completed professional job can provide:

1. Voice WAV for preview/editing
2. 48 kHz stereo Video Master WAV
3. SRT captions
4. VTT captions

The server does not add music.

## Studio features

- Chatterbox Turbo default, Original available from the same model selector
- Audio 1 by default; removable Audio 2–5 workspaces
- Sequential GPU queue with Generate All and Queue Monitor
- Five bundled predefined American male references plus uploaded clone references
- Reference-quality status in the UI
- Waveform, playhead, zoom, pan and audio cutter
- FastAPI documentation at `/docs`
- Responsive light/dark interface

## Google Colab

Open `colab/SoftMeta_Chatterbox_TTS_Colab_v1.5.0.ipynb` and run all cells. The notebook requests a GPU runtime with an L4 preference through Colab metadata. Actual L4 allocation remains controlled by Google Colab availability and account access.

The notebook reuses the main Python environment and cached model files on later runs unless `FORCE_REINSTALL` is enabled.

## Docker

Build with `docker compose up --build`. Generated audio and voice/reference folders are mounted as configured in `docker-compose.yml`.

## Licence

SoftMeta server/UI code is MIT licensed. See `THIRD_PARTY_NOTICES.md` for upstream components.
