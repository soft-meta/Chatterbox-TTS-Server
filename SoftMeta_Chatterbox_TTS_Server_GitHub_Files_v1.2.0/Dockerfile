FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 sox ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN python -m pip install --no-cache-dir -r requirements.txt
EXPOSE 8004
CMD ["python", "start.py"]
