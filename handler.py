import runpod
import os
import torch
from huggingface_hub import hf_hub_download
from pathlib import Path

# -----------------------------------------------------
# 1. Load WAN model (download if missing)
# -----------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_REPO = "Wan-AI/Wan2.2-Image2Video-14B"
MODEL_FILE = "model.safetensors"
CACHE_DIR = "/cache"

WAN_MODEL_PATH = None
WAN_MODEL = None


def load_model():
    global WAN_MODEL_PATH, WAN_MODEL

    if WAN_MODEL is not None:
        return WAN_MODEL

    print("🔍 Checking WAN model in cache...")

    WAN_MODEL_PATH = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        cache_dir=CACHE_DIR,
        token=HF_TOKEN
    )

    print("📦 WAN model located at:", WAN_MODEL_PATH)

    # --------------------------------------------------
    # LOAD MODEL + OPTIMIZE
    # --------------------------------------------------
    print("⚙️ Loading model into GPU...")
    model = torch.load(WAN_MODEL_PATH, map_location="cuda")
    model.eval()

    WAN_MODEL = model
    print("🚀 Model ready!")

    return WAN_MODEL


# -----------------------------------------------------
# 2. Inference Function
# -----------------------------------------------------

def generate_video(prompt: str, num_frames: int = 48):
    model = load_model()

    # NOTE:
    # WAN2.x inference code may differ depending on the repo.
    # Replace below logic with the official forward() call.

    with torch.no_grad():
        output = model.generate_video(
            prompt=prompt,
            num_frames=num_frames,
            guidance_scale=7.5,
            seed=42
        )

    video_path = "/tmp/output.mp4"
    output.save(video_path)

    return video_path


# -----------------------------------------------------
# 3. Runpod Handler Function
# -----------------------------------------------------

def handler(job):
    """Runpod job handler."""
    inp = job["input"]

    prompt = inp.get("prompt", "A cinematic waterfall flowing through neon rocks")
    num_frames = int(inp.get("num_frames", 48))

    print(f"🧪 Generating: {prompt} ({num_frames} frames)")

    output_video = generate_video(prompt, num_frames)

    return {
        "video_url": runpod.serverless.upload_file(output_video),
        "status": "success"
    }


# -----------------------------------------------------
# 4. Start Runpod Serverless
# -----------------------------------------------------

runpod.serverless.start({"handler": handler})
