ENV DEBIAN_FRONTEND=noninteractive
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y git wget unzip ffmpeg

WORKDIR /workspace

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download WAN 2.2 Image2Video 14B
RUN mkdir -p /models \
 && wget -O /models/model.safetensors \
    https://huggingface.co/Wan-AI/Wan2.2-Image2Video-14B/resolve/main/model.safetensors

COPY handler.py .

CMD ["python", "handler.py"]
