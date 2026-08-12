# SoftMeta Chatterbox TTS Server

## v1.6.1 Turbo Motivational Controls

v1.6.1 keeps the clean Chatterbox Turbo generation architecture from v1.6.0 and restores the creator controls requested from v1.2.1. No Auto Emotion, ASR/QC, speaker verification, retry/rescue, pronunciation rewrite, prosody stack, EQ/compression or advanced mastering has been reintroduced.

### Presets

- **Motivational Speech** is the default for calm senior-advisor, tutorial, health and life-advice narration. It restores the v1.2.1 Turbo profile: Temperature 0.72, Top P 0.90, Top K 1000, Repetition Penalty 1.20, Speed Factor 0.93 and a 140 ms chunk breath.
- **Chatterbox Turbo Default** restores the official-style Turbo sampling baseline: Temperature 0.8, Top P 0.95, Top K 1000, Repetition Penalty 1.20, Speed Factor 1.0 and an 80 ms chunk breath.

The preset fills the controls first; manual changes are honored by the backend.

### Turbo controls

Working controls: Temperature, Top P, Top K, Repetition Penalty, Speed Factor and Seed. Speed Factor is applied once with FFmpeg `atempo`, preserving pitch. Exaggeration and CFG Weight remain visible as Original-only references and are disabled because Turbo ignores them.

Turbo is English-only in this build. Voice identity, maturity and accent are inherited primarily from the reference clip, so use a clean American-English reference recording when an American accent is desired.

### Long-form reliability

Long scripts are split into sentence-safe chunks capped near 300 characters. Each chunk is generated exactly once. The only final audio processing is the optional Speed Factor pass plus the existing loudness-only normalization requested by the user.

### Colab

The v1.6.1 Colab notebook keeps the server alive under the existing supervisor and retains the manual Disconnect Colab control.
