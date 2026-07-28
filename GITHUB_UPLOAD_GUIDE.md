# GitHub upload guide

## 1. Create two empty repositories

Under the `soft-meta` GitHub organisation create:

- `chatterbox-v2`
- `Chatterbox-TTS-Server`

Do not initialise them with another README or licence because the folders already contain those files.

## 2. Upload `chatterbox-v2` first

Run the engine commands from the root README, create the `v0.1.0` tag, and verify the repository opens correctly.

## 3. Upload the server

Push `Chatterbox-TTS-Server`, then create its `v0.1.0` tag.

## 4. Test in Colab

Open `colab/Soft_Meta_Chatterbox_TTS_Colab.ipynb`. The notebook uses the two GitHub repositories and does not clone Devnen.

## 5. Release policy

Keep Colab pinned to release tags. Test changes on a `development` branch, then publish `v0.1.1`, `v0.2.0`, and so on. Avoid installing directly from a changing `main` branch for production use.
