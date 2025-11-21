FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y \
    git wget unzip ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install HuggingFace Hub for runtime download
RUN pip install --no-cache-dir huggingface_hub

COPY handler.py .

CMD ["python", "handler.py"]
