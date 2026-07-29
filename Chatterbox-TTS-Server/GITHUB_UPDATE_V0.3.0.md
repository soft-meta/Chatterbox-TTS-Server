# Update SoftMeta Chatterbox TTS Server to v0.3.0

The engine repository remains on `soft-meta/chatterbox-v2@v0.2.1`.
Only the server/UI repository needs the v0.3.0 upload and release.

## Upload

1. Open `soft-meta/Chatterbox-TTS-Server` on GitHub.
2. Upload the contents of the v0.3.0 server ZIP to the repository root.
3. Replace files when GitHub shows existing names.
4. Commit to `main` with:

```text
SoftMeta Chatterbox TTS Server v0.3.0
```

## Release

Create a normal release from `main`:

```text
Tag: v0.3.0
Title: SoftMeta Chatterbox TTS Server v0.3.0
```

Paste the contents of `RELEASE_NOTES_V0.3.0.md` into the release description.

## Colab

Upload `colab/SoftMeta_Chatterbox_TTS_Colab_v0.3.0.ipynb` to Google Colab,
select an L4 GPU, restart the runtime, and run all cells.
