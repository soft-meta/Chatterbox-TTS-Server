FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m venv --system-site-packages /opt/softmeta-voice \
    && /opt/softmeta-voice/bin/python -m pip install --no-cache-dir -U pip wheel setuptools \
    && /opt/softmeta-voice/bin/python -m pip install --no-cache-dir -r requirements-voice.txt
ENV SOFTMETA_VOICE_PYTHON=/opt/softmeta-voice/bin/python
EXPOSE 8004
CMD ["python", "start.py"]
