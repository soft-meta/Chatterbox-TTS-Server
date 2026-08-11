# SoftMeta Chatterbox TTS Server

## v1.5.10 Short-Tail QC Safety

Chatterbox Turbo remains the production default. The single **Generate Audio** workflow keeps the professional speech pipeline from v1.5.8, including pronunciation preparation, Senior Clear Speech, optional native Auto Emotion, natural pause shaping, age-aware pacing, dynamics-preserving mastering, Prosody reporting, reliable downloads, 48 kHz video master and SRT/VTT captions.

### Faster professional generation

The expensive Production QC architecture has been redesigned for Turbo long-form generation. Every generated chunk still receives an objective acoustic safety check, but Whisper ASR and SpeechBrain speaker verification no longer run between every TTS model call. The complete mastered narration receives one final ASR verification and one representative speaker-consistency check instead.

When faster-whisper supports its batched pipeline, final ASR uses a single batched pass with beam size 1 and word timestamps. If batched inference is unavailable, the server falls back to one ordinary final-file transcription rather than returning to per-chunk ASR.

Soft ASR disagreements no longer cause repeated Turbo re-generation. An objectively broken chunk may receive one focused retry; the existing smaller-chunk rescue remains available only for a genuine hard acoustic failure.

Turbo-native event sentences are now packed with nearby narration instead of forcing one tiny model call for every `[chuckle]`, `[laugh]`, `[sigh]`, or `[gasp]`. Event-bearing spans remain moderately local while reducing long-form model-call overhead.


### Short-tail false-failure fix

Fast Professional QC no longer treats speaking rate by itself as proof of corrupt audio. Short final chunks and Turbo event spans can legitimately contain longer pauses or non-verbal events, making words-per-minute unreliable. Only objective waveform failures such as empty audio, invalid samples, unusable level or mostly-silent audio can hard-stop a Turbo job. Speaking-rate outliers remain advisory, and short spans are exempt from normal rate-drift retries unless timing is truly absurd.

### Optional Auto Emotion

Auto Emotion is **OFF by default**. When enabled, only four audible Turbo-native events are inserted when the wording genuinely supports them: `[chuckle]`, `[laugh]`, `[sigh]`, and `[gasp]`. Neutral text receives no fabricated event.

### Queue and downloads

Finished Queue Monitor cards can be dismissed without deleting their generated audio. The minimized progress card remains draggable and shows live completion progress. Final WAVs, cuts and platform assets use server-enforced attachment downloads for reliable delivery through the Colab proxy.

### Original model

Chatterbox Original remains selectable and keeps its previous generation/QC behavior. The v1.5.10 fast final-pass architecture is enabled only for the advanced Chatterbox Turbo workflow.
