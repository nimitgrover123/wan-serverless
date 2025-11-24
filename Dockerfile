FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=1

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y git wget unzip ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-build-isolation --no-cache-dir -r requirements.txt

COPY . .

COPY handler.py .

CMD ["python", "handler.py"]
