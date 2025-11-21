FROM runpod/pytorch:2.1.0-py3.10-cuda12.1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
COPY runpod.yaml .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
