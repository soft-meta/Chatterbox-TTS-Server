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

## LongCat Video Avatar 1.5

Avatar Talking invokes LongCat's official single-audio image-to-video pipeline
in an isolated Python 3.10 environment. SoftMeta selectively downloads the
official base and Avatar 1.5 components needed for INT8 distilled inference.

- Upstream: https://github.com/meituan-longcat/LongCat-Video
- Avatar model: https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5
- Base model: https://huggingface.co/meituan-longcat/LongCat-Video
- Licence: MIT

SoftMeta pins upstream revisions and applies a guarded operational patch to the
official entrypoint. The patch makes the seed configurable and avoids repeated
intermediate continuation encodes; it does not replace the model or claim
ownership of LongCat code or weights.

## FFmpeg

SoftMeta uses the locally installed FFmpeg executable for audio conversion,
section joining, final MP4 encoding and technical media checks. FFmpeg licensing
depends on the build installed by the user.

## SoftMeta code

The server, queues, browser UI, waveform tools, audio cutter, voice workflow,
avatar orchestration, persistent job metadata and quality checks are maintained
by SoftMeta under the repository MIT licence.
