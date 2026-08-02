# SoftMeta v0.7.0 GitHub update

Update only `soft-meta/Chatterbox-TTS-Server`. Keep `soft-meta/chatterbox-v2@v0.2.1`.

## Upload

Extract `soft-meta-Chatterbox-TTS-Server-v0.7.0.zip` and upload the contents of the `Chatterbox-TTS-Server` folder to the repository root, replacing existing files.

## Commit

Commit message:

`Add identity-first unique voice search v0.7.0`

Extended description:

`Added strict Qwen over-generation, pairwise speaker verification, automatic repeated-identity rejection, truthful similarity labels and stronger age-separated human voice profiles.`

## Release

- Tag: `v0.7.0`
- Target: `main`
- Title: `SoftMeta Chatterbox TTS Server v0.7.0`
- Do not mark as a pre-release

Paste `RELEASE_NOTES_V0.7.0.md` into the release description.

After publishing the release, open `colab/SoftMeta_Chatterbox_TTS_Colab_v0.7.0.ipynb` in a fresh L4 runtime and run all cells.
