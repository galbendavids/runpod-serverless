FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Whisper models during the build process
RUN python3 -c 'import faster_whisper; m1 = faster_whisper.WhisperModel("whisper-large-v3"); m2 = faster_whisper.WhisperModel("ivrit-ai/faster-whisper-v2-d4")'

COPY infer.py .

ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

CMD ["python", "-u", "infer.py"]