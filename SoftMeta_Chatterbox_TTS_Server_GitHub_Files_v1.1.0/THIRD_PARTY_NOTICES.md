# Third-party notices

## Chatterbox

- Upstream: https://github.com/resemble-ai/chatterbox
- Licence: MIT

## MOSS VoiceGenerator

Generate Voice uses MOSS VoiceGenerator to create fictional reference voices
from free-form descriptions without reference audio.

- Upstream: https://github.com/OpenMOSS/MOSS-TTS
- Model: https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator
- Licence: Apache-2.0

## SpeechBrain speaker embeddings

Candidate identity screening uses SpeechBrain ECAPA-TDNN embeddings. Similarity
screening is an aid, not a guarantee that every listener will perceive two
samples as different people.

- Upstream: https://github.com/speechbrain/speechbrain
- Model: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- Licence: Apache-2.0

## FFmpeg

SoftMeta uses the locally installed FFmpeg executable for audio conversion,
waveform processing and audio cutting. FFmpeg licensing depends on the build
installed by the user.

## SoftMeta code

The server, audio queue, browser UI, waveform tools, audio cutter and voice
workflow are maintained by SoftMeta under the repository MIT licence.
