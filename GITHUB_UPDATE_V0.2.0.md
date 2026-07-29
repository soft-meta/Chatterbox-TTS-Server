# Update existing SoftMeta repositories to v0.2.0

Do not edit the old `v0.1.0` releases. Releases are permanent snapshots.

## 1. Update `soft-meta/chatterbox-v2`

Upload the contents of the v0.2.0 engine ZIP to the repository root, replacing
matching files. Commit message:

```text
SoftMeta chatterbox-v2 v0.2.0
```

Create a release tag `v0.2.0` from `main`.

## 2. Update `soft-meta/Chatterbox-TTS-Server`

Upload the contents of the v0.2.0 server ZIP to the repository root, replacing
matching files. Commit message:

```text
SoftMeta Chatterbox TTS Server v0.2.0
```

Create a release tag `v0.2.0` from `main`.

## 3. Test in Colab

Upload `colab/SoftMeta_Chatterbox_TTS_Colab_v0.2.0.ipynb` to Colab, select L4,
and run all cells. Use the log cell if model loading fails.
