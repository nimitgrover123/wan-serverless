FROM runpod/base:0.6.0-cuda12.1

RUN apt update && apt install -y git wget unzip ffmpeg

WORKDIR /workspace

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download WAN 2.2 Image2Video 14B model
RUN mkdir -p /models \
    && wget -O /models/i2v14b.safetensors https://huggingface.co/Wan-AI/Wan2.2-Image2Video-14B/resolve/main/model.safetensors

COPY handler.py .
COPY wan_start.sh .

CMD ["/bin/bash", "wan_start.sh"]
