# Third-party notices

## Chatterbox

SoftMeta Chatterbox TTS Server uses the official open-source Chatterbox package
and model technology developed by Resemble AI.

- Upstream: https://github.com/resemble-ai/chatterbox
- Licence: MIT

## MOSS VoiceGenerator

The optional **Generate Voice** feature uses MOSS VoiceGenerator to create new
fictional reference voices from free-form descriptions without reference audio.

- Upstream: https://github.com/OpenMOSS/MOSS-TTS
- Model: OpenMOSS-Team/MOSS-VoiceGenerator
- Licence: Apache-2.0

## SpeechBrain speaker embeddings

The candidate identity check uses SpeechBrain ECAPA-TDNN speaker embeddings.
It is a similarity aid, not a guarantee that two voices will sound different to
every listener.

- Upstream: https://github.com/speechbrain/speechbrain
- Model: speechbrain/spkrec-ecapa-voxceleb
- Licence: Apache-2.0

## SoftMeta code

The server, queue, web UI, waveform tools, audio cutter, voice-candidate workflow,
quality screening and Colab launcher are maintained by SoftMeta under the
repository MIT licence.
