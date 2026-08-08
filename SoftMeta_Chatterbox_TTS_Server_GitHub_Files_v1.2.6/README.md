# SoftMeta Chatterbox TTS Server

A self-hosted Chatterbox TTS studio maintained by **SoftMeta**.

## Studio features

- Chatterbox Turbo is the default English model
- Motivational Speech is tuned for calm, naturally slow senior-advisor narration
- Serious Senior Advisor Auto Emotion for Turbo: the whole script is scanned first, headings are protected, and only sparse high-value expressive cues are added
- Auto Emotion never auto-inserts laugh, chuckle, angry, crying, sarcastic or audible sigh cues in Motivational Speech
- Turbo Excessive Silence Guard compresses multi-second generated dead air to a clean natural breath pause while preserving normal short pauses
- Turbo Motivational Speech locks the server path to English and normalizes punctuation for steadier American-English narration; cloned accent still depends on the reference voice
- Per-chunk Turbo stability checks, level matching and final loudness mastering
- Audio 1 by default; add removable Audio 2–5 workspaces
- Sequential audio queue with Generate All and Queue Monitor
- Five bundled predefined American male voices plus reference voice cloning
- Main waveform, live playhead, zoom, pan and audio cutter
- Download full audio, selected audio, Part One or Part Two
- FastAPI documentation at `/docs`
- Responsive light and dark interface

## Google Colab

Open `colab/SoftMeta_Chatterbox_TTS_Colab_v1.2.6.ipynb` and run all cells.
The notebook requests a GPU runtime with L4 preference through Colab metadata. Actual L4 allocation remains controlled by Google Colab availability and account access.

The notebook reuses verified environments and downloaded model files on later runs.

## Docker

Build with `docker compose up --build`. Generated audio, reference voices and predefined voices are mounted from the local directories shown in `docker-compose.yml`.

## Licence

SoftMeta server and UI code are MIT licensed. Chatterbox is MIT licensed; see `THIRD_PARTY_NOTICES.md`.
