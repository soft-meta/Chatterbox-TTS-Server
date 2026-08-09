# SoftMeta Chatterbox TTS Server

Self-hosted long-form Chatterbox narration studio maintained by **SoftMeta**.

## v1.5.5 Turbo Avatar Performance

v1.5.5 makes **Chatterbox Turbo the production default** for the current senior-video workflow. Chatterbox Original remains available for comparison but its v1.5.4 behavior is intentionally left unchanged while the Turbo pipeline is tuned first.

### First 5-minute Avatar Performance Mode

- Turbo Auto Emotion is front-loaded into the first five minutes, where the narration is intended to drive a lip-synced talking avatar.
- A full five-minute avatar window targets **10 restrained native emotion beats** when the script is long enough.
- The planner prefers semantic emotion first, then fills only genuine long flat gaps with a calm `[narration]` reset.
- Emotion spacing is checked so the opening does not contain long 60–70 second flat pockets when a safe sentence is available.
- After the first five minutes, the plan becomes deliberately calmer for B-roll/natural-scene narration.
- Auto-generated tags stay limited to Turbo-supported serious-narration cues: `[happy]`, `[narration]`, `[surprised]`, and `[dramatic]`. Comedic, angry, crying, laughing and similar auto-tags remain disabled.

### Turbo-native emotion delivery

- Native emotion tokens are preserved all the way to Chatterbox Turbo inference.
- Emotion sentences are isolated into compact local speech segments instead of being buried in a large 70–85 word chunk.
- The first-five-minute emotion segments receive a wider adaptive pace band so post-processing does not force every emotional beat back to one cadence.
- Warm/surprised beats may stay slightly more alive; reflective/serious beats are allowed to slow naturally.
- ASR-only uncertainty is advisory and does not repeatedly regenerate a healthy expressive take until the emotion is flattened.

### Dynamic final mastering

Turbo uses a dynamics-preserving final mastering path:

- gentler compression than the standard professional path
- slower attack/release so short expressive movement survives
- approximately **-13 LUFS** final spoken-word target
- approximately **-1 dBTP** peak safety target
- chunk leveling avoids forcing every accepted segment to identical RMS
- long intentional pauses remain perceptibly longer than ordinary pauses after silence cleanup

The mastering stage does not fabricate emotion. Its job is to preserve the dynamic movement produced by Turbo while keeping the final file clear and consistent.

### Prosody Quality Advisor

Completed Turbo professional jobs also receive an advisory Prosody score. It reports:

- first-five-minute emotion beats versus target
- maximum estimated emotion gap in the avatar window
- measured final loudness range (LRA)
- first-five-minute short-term RMS movement
- longest acoustically flat stretch
- integrated loudness and true peak

This score is diagnostic only and never rejects an otherwise healthy job. Production QC continues to handle waveform integrity, ASR/script verification, retries and optional speaker consistency.

### Shared production features retained

- Senior Clear Speech text preparation
- Pronunciation Engine and editable `pronunciations.json`
- heading protection
- Intelligent Micro-Pause and excessive-silence cleanup
- age-aware target-band pacing for 60s, 70s, 80s and 90+ listeners
- section openings, intro pacing and retention structure
- reference voice quality analysis
- Production QC and failed-chunk rescue
- 48 kHz stereo Video Master WAV
- SRT and VTT captions
- optional performance feedback storage

## Chatterbox Original status

Original is intentionally **not part of the new v1.5.5 tuning experiment**. It remains selectable and keeps its previous v1.5.4 behavior and controls. Once Turbo is validated successfully on real Colab output, the proven improvements can be ported to Original separately.

## Studio features

- **Chatterbox Turbo default**, Original still selectable
- Audio 1 by default; removable Audio 2–5 workspaces
- sequential GPU queue with Generate All and Queue Monitor
- bundled predefined American male references plus uploaded clone references
- reference-quality status in the UI
- waveform, playhead, zoom, pan and audio cutter
- FastAPI documentation at `/docs`
- responsive light/dark interface

## Google Colab

Open `colab/SoftMeta_Chatterbox_TTS_Colab_v1.5.5.ipynb` and run all cells. The notebook requests a GPU runtime with an L4 preference through Colab metadata. Actual GPU allocation is controlled by Google Colab availability and account access.

The notebook starts the server with `SOFTMETA_MODEL=chatterbox-turbo` and reuses cached packages/model files on later runs unless `FORCE_REINSTALL` is enabled.

## Docker

Build with `docker compose up --build`. Generated audio and voice/reference folders are mounted as configured in `docker-compose.yml`.

## Licence

SoftMeta server/UI code is MIT licensed. See `THIRD_PARTY_NOTICES.md` for upstream components.
