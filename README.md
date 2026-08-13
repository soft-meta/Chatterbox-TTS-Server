# SoftMeta Chatterbox TTS Server

## v1.6.4 Turbo Creator Controls

Colab setup now checks the GitHub server version before the heavy Torch/Chatterbox install, accepts compatible v1.6.2+ servers within the v1.6 line instead of aborting on an exact-version mismatch, and verifies the installed Turbo generate() signature before startup. Micromamba still uses the official micro.mamba.pm endpoint first with the pinned GitHub release as fallback; partial or invalid downloads are discarded before reuse.

v1.6.4 keeps the clean Chatterbox Turbo generation architecture from v1.6.0 and restores the creator controls requested from v1.2.1. No Auto Emotion, ASR/QC, speaker verification, retry/rescue, pronunciation rewrite, prosody stack, EQ/compression or advanced mastering has been reintroduced.

### Presets

- **Motivational Speech** is the default for calm senior-advisor, tutorial, health and life-advice narration. It restores the v1.2.1 Turbo profile: Temperature 0.72, Top P 0.90, Top K 1000, Repetition Penalty 1.20, Speed Factor 0.93 and a 140 ms chunk breath.
- **Chatterbox Turbo Default** restores the official-style Turbo sampling baseline: Temperature 0.8, Top P 0.95, Top K 1000, Repetition Penalty 1.20, Speed Factor 1.0 and an 80 ms chunk breath.

The preset fills the controls first; manual changes are honored by the backend.

### Turbo controls

Working controls: Temperature, Top P, Top K, Repetition Penalty, Speed Factor, Seed, Exaggeration and CFG Weight. Because official Turbo ignores native CFG/exaggeration during inference, the server bridges those two creator sliders conservatively into supported Turbo sampling controls while keeping the clean generation path.

Turbo is English-only in this build. Voice identity, maturity and accent are inherited primarily from the reference clip, so use a clean American-English reference recording when an American accent is desired.

### Long-form reliability

Long scripts are split into sentence-safe chunks capped near 300 characters. Each chunk is generated exactly once. The only final audio processing is the optional Speed Factor pass plus the existing loudness-only normalization requested by the user.

### Colab

The v1.6.4 Colab notebook keeps the server alive under the existing supervisor and retains the manual Disconnect Colab control.
