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

## EchoMimicV3 Flash

Avatar Talking invokes Ant Group's official EchoMimicV3 Flash image-and-audio
pipeline in an isolated Python 3.10 environment.

- Upstream: https://github.com/antgroup/echomimic_v3
- Flash model: https://huggingface.co/BadToBest/EchoMimicV3
- Base model: https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP
- Audio encoder: https://huggingface.co/TencentGameMate/chinese-wav2vec2-base
- Licence: Apache-2.0 for EchoMimicV3; bundled dependencies retain their licences

SoftMeta pins upstream revisions and applies a guarded batch and heartbeat patch
to the official entrypoint. It does not replace the model or claim ownership of
EchoMimicV3 code or weights.

## FFmpeg

SoftMeta uses the locally installed FFmpeg executable for audio conversion,
section joining, final MP4 encoding and technical media checks. FFmpeg licensing
depends on the build installed by the user.

## SoftMeta code

The server, queues, browser UI, waveform tools, audio cutter, voice workflow,
avatar orchestration, persistent job metadata and quality checks are maintained
by SoftMeta under the repository MIT licence.
