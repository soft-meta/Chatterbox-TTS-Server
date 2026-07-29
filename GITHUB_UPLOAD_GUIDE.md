# GitHub update guide for v0.2.0

The `soft-meta/chatterbox-v2` and `soft-meta/Chatterbox-TTS-Server`
repositories already exist. Do not delete the old `v0.1.0` releases.

## Engine repository

1. Open `soft-meta/chatterbox-v2`.
2. Upload the contents of the v0.2.0 engine folder to the repository root.
3. Replace matching files.
4. Commit as `SoftMeta chatterbox-v2 v0.2.0`.
5. Create a normal release using tag `v0.2.0` from `main`.

## Server repository

1. Open `soft-meta/Chatterbox-TTS-Server`.
2. Upload the contents of the v0.2.0 server folder to the repository root.
3. Replace matching files.
4. Commit as `SoftMeta Chatterbox TTS Server v0.2.0`.
5. Create a normal release using tag `v0.2.0` from `main`.

## Colab

After both tags exist, upload
`colab/SoftMeta_Chatterbox_TTS_Colab_v0.2.0.ipynb` to Google Colab,
select an L4 GPU and run every cell.
