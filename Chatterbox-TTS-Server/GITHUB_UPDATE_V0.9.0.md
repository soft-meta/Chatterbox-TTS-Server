# SoftMeta v0.9.0 GitHub update

Update only `soft-meta/Chatterbox-TTS-Server`.
Keep `soft-meta/chatterbox-v2@v0.2.1` unchanged.

## Upload

Extract `soft-meta-Chatterbox-TTS-Server-v0.9.0.zip` and replace the repository
root with the contents of its `Chatterbox-TTS-Server` folder.

## Commit

Commit message:

`Add long-form Avatar Talking and Generate Video v0.9.0`

Extended description:

`Added the moving Generate Video workspace, A100 Ditto worker, long-video queue, direct Audio 1–5 source selection, avatar and audio uploads, persistent progress, MP4 preview and download, checkpointed rendering, and technical quality checks.`

## Release

- Tag: `v0.9.0`
- Target: `main`
- Title: `SoftMeta Chatterbox TTS Server v0.9.0`
- Do not mark as a pre-release
- Paste `RELEASE_NOTES_V0.9.0.md` into the release description

After publishing, open
`colab/SoftMeta_Chatterbox_TTS_Colab_v0.9.0.ipynb`, select A100 and run all
cells. Begin with a 10–20 second test before generating a long video.
