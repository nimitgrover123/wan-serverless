import runpod
import torch
from diffusers import DiffusionPipeline
import base64
import os
import tempfile
from huggingface_hub import hf_hub_download
import os

HF_TOKEN = os.getenv("HF_TOKEN")

model_path = hf_hub_download(
    repo_id="Wan-AI/Wan2.2-Image2Video-14B",
    filename="model.safetensors",
    cache_dir="/cache",
    token=HF_TOKEN
)

print("Loaded WAN model at:", model_path)
MODEL_ID = "DiffusionCraft/wan-2.2-image2video-14b-lite"

pipe = None

def load_model():
    global pipe
    if pipe is None:
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
    return pipe

def run(job):
    req = job["input"]

    image_b64 = req["image"]
    seconds = req.get("seconds", 2)
    fps = req.get("fps", 24)

    image_bytes = base64.b64decode(image_b64)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
        f.write(image_bytes)
        input_image_path = f.name

    pipe = load_model()

    video = pipe(
        image=input_image_path,
        num_frames=seconds * fps,
    ).frames

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    video.write_videofile(output_path, fps=fps)

    with open(output_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "video_base64": video_b64,
        "seconds": seconds,
        "fps": fps
    }

runpod.serverless.start({"handler": run})
