# Update SoftMeta Chatterbox TTS Server to v0.3.1

Only the server/UI repository needs an update. Keep the engine at v0.2.1.

## Commit

Commit message:

`Fix isolated Generate Voice environment v0.3.1`

Extended description:

`Separated Parler-TTS from Chatterbox dependencies and added a dedicated voice worker for reliable Colab and Docker generation.`

## Release

- Tag: `v0.3.1`
- Target: `main`
- Title: `SoftMeta Chatterbox TTS Server v0.3.1`
- Paste `RELEASE_NOTES_V0.3.1.md` into the release description
- Do not mark it as a pre-release

## Colab

After publishing the release, upload `colab/SoftMeta_Chatterbox_TTS_Colab_v0.3.1.ipynb` to a fresh L4 runtime and use Run all.
