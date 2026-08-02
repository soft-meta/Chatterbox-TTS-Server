# Third-party notices

## Chatterbox

SoftMeta uses the official open-source Chatterbox package and model technology
developed by Resemble AI.

- Upstream: https://github.com/resemble-ai/chatterbox
- Licence: MIT

## MOSS VoiceGenerator

Generate Voice uses MOSS VoiceGenerator to create fictional reference voices
from free-form descriptions without reference audio.

- Upstream: https://github.com/OpenMOSS/MOSS-TTS
- Model: OpenMOSS-Team/MOSS-VoiceGenerator
- Licence: Apache-2.0

## SpeechBrain speaker embeddings

Candidate identity screening uses SpeechBrain ECAPA-TDNN speaker embeddings. It
is a similarity aid, not a guarantee that two samples will sound different to
every listener.

- Upstream: https://github.com/speechbrain/speechbrain
- Model: speechbrain/spkrec-ecapa-voxceleb
- Licence: Apache-2.0

## Ditto TalkingHead

Avatar Talking invokes the official Ditto `inference.py` in an isolated Python
3.10 environment.

- Upstream: https://github.com/antgroup/ditto-talkinghead
- Checkpoints: https://huggingface.co/digital-avatar/ditto-talkinghead
- Code licence: Apache-2.0

### Checkpoint licensing warning

The official Ditto checkpoint bundle contains third-party face-analysis files,
including `insightface_det.onnx`, `insightface_det_fp16.engine` and/or
`det_10g.onnx`. InsightFace states that the pretrained models it supplies are
available for non-commercial research purposes unless separately licensed.
SoftMeta does not grant rights to those assets. Review and satisfy all relevant
third-party terms, or replace those detector assets, before commercial or
monetized use.

- InsightFace licence notice: https://github.com/deepinsight/insightface#license

## FFmpeg

SoftMeta uses a locally installed FFmpeg executable for audio conversion,
section joining, final MP4 encoding and technical media checks. FFmpeg licensing
depends on the build installed by the user.

## SoftMeta code

The server, queues, browser UI, waveform tools, audio cutter, voice workflow,
avatar orchestration, persistent job metadata and quality checks are maintained
by SoftMeta under the repository MIT licence.
