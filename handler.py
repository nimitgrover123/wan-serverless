import runpod
import os
import torch
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_REPO = "Wan-AI/Wan2.2-Image2Video-14B"
CACHE_DIR = "/cache"

pipe = None


def load_model():
    global pipe
    if pipe:
        return pipe

    print("⬇️ Downloading WAN 2.2 Model...")

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_REPO,
        cache_dir=CACHE_DIR,
        token=HF_TOKEN,
        torch_dtype=torch.float16
    ).to("cuda")

    print("🚀 WAN 2.2 Loaded!")
    return pipe


def generate_video(prompt: str, num_frames: int = 48):
    pipe = load_model()

    print("🎬 Generating video...")

    output = pipe(
        prompt=prompt,
        num_frames=num_frames,
        guidance_scale=7.5
    )

    video = output.videos[0]

    out_path = "/tmp/output.mp4"
    video.save(out_path)

    return out_path


def handler(job):
    inp = job["input"]
    prompt = inp.get("prompt", "A neon waterfall in a glowing forest")
    num_frames = int(inp.get("num_frames", 48))

    print(f"⚡ Job received: {prompt}")

    video_path = generate_video(prompt, num_frames)

    return {
        "status": "success",
        "video_url": runpod.serverless.upload_file(video_path)
    }


runpod.serverless.start({"handler": handler})
