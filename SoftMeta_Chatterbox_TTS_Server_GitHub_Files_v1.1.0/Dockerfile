FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 sox ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m venv /opt/softmeta-moss \
    && /opt/softmeta-moss/bin/python -m pip install --no-cache-dir -U pip wheel setuptools \
    && git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS.git /opt/MOSS-TTS \
    && /opt/softmeta-moss/bin/python -m pip install --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cu128 \
       -e "/opt/MOSS-TTS[torch-runtime]" \
    && /opt/softmeta-moss/bin/python -m pip install --no-cache-dir -r requirements-voice.txt
ENV SOFTMETA_VOICE_PYTHON=/opt/softmeta-moss/bin/python
ENV SOFTMETA_MOSS_MODEL_DIR=/models/moss_voice_generator
# Avatar Talking uses a separately prepared Python 3.10 EchoMimicV3 environment.
ENV SOFTMETA_AVATAR_PYTHON=/opt/softmeta-avatar/bin/python
ENV SOFTMETA_ECHOMIMIC_DIR=/opt/EchoMimicV3
ENV SOFTMETA_ECHOMIMIC_MODELS=/models/echomimic
EXPOSE 8004
CMD ["python", "start.py"]
