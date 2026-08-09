# SoftMeta Chatterbox TTS Server

Self-hosted long-form Chatterbox narration studio maintained by **SoftMeta**.

## v1.5.6 Turbo Standard vs Advanced Human Performance

v1.5.6 keeps **Chatterbox Turbo as the production default** and adds a true A/B workflow. Chatterbox Original remains available but its existing behaviour is intentionally left unchanged while Turbo is validated first.

### Generate Audio: clean Turbo baseline

The first button is deliberately simple so it can be used as a control sample:

- one direct Chatterbox Turbo model call for short A/B scripts
- creator text is passed to Turbo unchanged
- official-style Turbo sampling defaults: temperature 0.8, top-p 0.95, top-k 1000, repetition penalty 1.2, min-p 0
- no SoftMeta Auto Emotion
- no Senior Clear Speech rewrite
- no advanced chunk planner, adaptive pacing, Production QC rerendering, final mastering, captions or platform assets
- the model's own loudness normalisation remains active inside the installed Chatterbox engine

This path is intended for short controlled comparisons such as the same one-minute script and reference voice used with both buttons.

### Generate Advanced Audio: Turbo Human Performance

The second button keeps the professional pipeline but fixes the old emotion architecture. Previous builds inserted abstract tags such as `[happy]`, `[narration]`, `[surprised]` and `[dramatic]`; those did not reliably create audible human actions. v1.5.6 instead plans the Turbo event controls demonstrated by the upstream Turbo UI, including `[laugh]`, `[chuckle]`, `[sigh]` and `[gasp]`.

A critical v1.5.5 bug was also fixed: the Senior Clear Speech punctuation normaliser used to remove the square brackets from new event tags before inference, turning `[chuckle]` into the literal word `chuckle`. v1.5.6 protects Turbo event tokens through pronunciation and text preparation so the bracketed control token reaches the Turbo engine intact.

Automatic events are semantic rather than periodic:

- `[laugh]` for strong explicit laughter/hilarious moments
- `[chuckle]` for lighter laughter, smiling or wry moments
- `[sigh]` for regret, loss, loneliness and hard lessons
- `[gasp]` for explicit surprise or an unexpected reveal
- manual buttons are available for `[chuckle]`, `[laugh]`, `[sigh]`, `[gasp]` and `[clear throat]`
- ordinary serious advice does not receive a fake laugh or gasp simply to hit a quota
- the first five-minute avatar window is prioritised when valid emotional moments exist, while later B-roll narration stays calmer

### Advanced professional pipeline retained

Advanced Audio still includes:

- pronunciation preparation
- Senior Clear Speech punctuation and phrasing
- emotion-aware compact segmentation
- Intelligent Micro-Pause and excessive-silence cleanup
- age-aware target-band pacing for 60s, 70s, 80s and 90+ listeners
- section openings, intro pacing and retention structure
- reference voice quality analysis
- Production QC with advisory ASR and failed-chunk rescue
- dynamics-preserving Turbo mastering
- Prosody Quality Advisor
- 48 kHz stereo Video Master WAV
- SRT and VTT captions
- optional performance feedback storage

The A/B panel keeps the latest completed Standard and Advanced files side by side so the same script and voice can be compared directly.

## Chatterbox Original status

Original is intentionally **not part of the new v1.5.6 Turbo tuning experiment**. It remains selectable and keeps its previous v1.5.4 behavior and controls. Once Turbo is validated successfully on real Colab output, the proven improvements can be ported to Original separately.

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

Open `colab/SoftMeta_Chatterbox_TTS_Colab_v1.5.6.ipynb` and run all cells. The notebook requests a GPU runtime with an L4 preference through Colab metadata. Actual GPU allocation is controlled by Google Colab availability and account access.

The notebook starts the server with `SOFTMETA_MODEL=chatterbox-turbo` and reuses cached packages/model files on later runs unless `FORCE_REINSTALL` is enabled.

## Docker

Build with `docker compose up --build`. Generated audio and voice/reference folders are mounted as configured in `docker-compose.yml`.

## Licence

SoftMeta server/UI code is MIT licensed. See `THIRD_PARTY_NOTICES.md` for upstream components.
