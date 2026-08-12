# SoftMeta Chatterbox TTS Server

## v1.6.0 Fresh Turbo Reset

v1.6.0 removes the experimental professional narration stack and restores a direct Chatterbox Turbo workflow for long-form voice generation.

### Generate Audio

Every Generate Audio job uses Chatterbox Turbo with the Turbo sampling defaults used by the upstream demo:

- temperature `0.8`
- top-p `0.95`
- top-k `1000`
- repetition penalty `1.2`
- min-p `0.0`
- random seed per job/chunk

The server does not run Auto Emotion, pronunciation rewriting, Senior Clear Speech, age pacing, tempo changes, Production QC, Whisper ASR, speaker verification, retry/rescue generation, Prosody processing, EQ, compression, de-essing, caption generation or video-master generation.

### Long scripts

Turbo is a short-input model, so long scripts are divided only at sentence-safe boundaries into calls of at most 300 characters. A text-integrity check runs before inference. If the splitter ever changes the lexical script content, the job stops before spending GPU time.

Each safe chunk is sent to Turbo exactly once. There is no quality-warning retry loop.

### Output volume

The only final post-processing is a two-pass linear FFmpeg loudness normalization targeting approximately `-12.5 LUFS` with a `-0.8 dB` true-peak ceiling. This keeps the stronger output level from recent builds without applying the former professional processing chain.

### Colab runtime

The v1.6.0 Colab notebook starts the server under a keep-alive supervisor. While the Colab runtime is alive, an unexpected child-server exit is restarted automatically. The Audio Studio also exposes a manual **Disconnect Colab** control when running inside Colab. After disconnecting the runtime, reconnect from the Colab page.

Google Colab can still end a runtime because of its own session or resource policies; the server does not contain any job-completion auto-disconnect behavior.

### Preserved workflow

Voice cloning/predefined voices, Audio 1–5 queueing, Queue Monitor, waveform preview/cutting, reliable attachment downloads, draggable minimized progress, Remove All and individual Queue Monitor history dismissal remain available.
