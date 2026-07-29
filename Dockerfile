FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 sox && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m venv /opt/softmeta-qwen \
    && /opt/softmeta-qwen/bin/python -m pip install --no-cache-dir -U pip wheel setuptools \
    && /opt/softmeta-qwen/bin/python -m pip install --no-cache-dir -r requirements-voice.txt
ENV SOFTMETA_VOICE_PYTHON=/opt/softmeta-qwen/bin/python
EXPOSE 8004
CMD ["python", "start.py"]
