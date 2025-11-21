FROM runpod/serverless-cuda:12.1

# Install dependencies
RUN apt update && apt install -y git wget unzip ffmpeg

WORKDIR /workspace

# Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download WAN 2.2 Image2Video 14B model
RUN mkdir -p /models \
    && wget -O /models/model.safetensors \
       https://huggingface.co/Wan-AI/Wan2.2-Image2Video-14B/resolve/main/model.safetensors

COPY handler.py .

CMD ["python", "handler.py"]
