import os
import runpod
import torch
from pathlib import Path
import imageio

# ---------------------------------------------------------
# WAN 2.2 LOCAL MODEL SETUP
# ---------------------------------------------------------

MODEL_PATH = "/workspace/models/wan2.2_i2v_14b.safetensors"

WAN_MODEL = None


# Dummy WAN loader — replace with official code
class WAN22_I2V:
    def __init__(self, ckpt):
        print("Loading WAN 2.2 I2V model...")
        self.model = torch.load(ckpt, map_location="cuda")
        self.model.eval()
        print("WAN ready.")

    @torch.no_grad()
    def infer(self, prompt, num_frames=48):
        # Replace with official WAN inference call
        print(f"Running WAN inference for: {prompt}")

        # Fake output frames for structure (replace)
        frames = [(255 * torch.rand(256, 256, 3)).byte().cpu().numpy()
                  for _ in range(num_frames)]
        return frames


# ---------------------------------------------------------
# Load model once per container
# ---------------------------------------------------------

def load_model():
    global WAN_MODEL

    if WAN_MODEL is not None:
        return WAN_MODEL

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")

    WAN_MODEL = WAN22_I2V(MODEL_PATH)
    return WAN_MODEL


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

def generate_video(prompt, num_frames):
    model = load_model()

    frames = model.infer(prompt, num_frames)

    output_path = "/tmp/output.mp4"
    imageio.mimsave(output_path, frames, fps=24)

    return output_path


# ---------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------

def handler(job):
    inp = job.get("input", {})

    prompt = inp.get("prompt", "A cinematic neon waterfall")
    num_frames = int(inp.get("num_frames", 48))

    print(f"Generating video: {prompt} ({num_frames} frames)")

    video_path = generate_video(prompt, num_frames)

    url = runpod.serverless.upload_file(video_path)

    return {
        "status": "success",
        "video_url": url
    }


runpod.serverless.start({"handler": handler})
