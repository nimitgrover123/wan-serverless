"""
RunPod Serverless handler for WAN 2.2 (Image->Video) using a RunPod Volume for model files.

Expectations:
 - RunPod Volume mounted at /runpod-volume
 - WAN checkpoint files placed under /runpod-volume/wan2.2-i2v/
 - Your image input can be a path on the volume or a base64 data URI

This handler runs the WAN repo's generate.py as a subprocess from a repo you place in the image
or clone at runtime. If you prefer an in-process call, replace run_generate() with direct imports.
"""

import os
import shlex
import subprocess
import uuid
import time
import base64
import traceback
from pathlib import Path

import runpod

# Config (override via environment variables in RunPod UI if needed)
WAN_REPO_DIR = os.environ.get("WAN_REPO_DIR", "/workspace/Wan2.2-I2V-A14B")
RUNPOD_VOLUME_MOUNT = os.environ.get("RUNPOD_VOLUME_MOUNT", "/workspace")
CKPT_SUBDIR = os.environ.get("CKPT_SUBDIR", "Wan2.2-I2V-A14B")
CKPT_DIR = os.environ.get("CKPT_DIR", f"{RUNPOD_VOLUME_MOUNT}/{CKPT_SUBDIR}")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/wan_out")
GENERATE_PY = os.path.join(WAN_REPO_DIR, "generate.py")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper: verify world
def ensure_model_and_repo():
    if not Path(CKPT_DIR).exists():
        raise FileNotFoundError(
            f"Checkpoint dir not found on mounted volume: {CKPT_DIR}. Upload model files to the RunPod volume and mount it at {RUNPOD_VOLUME_MOUNT}")
    if not Path(GENERATE_PY).exists():
        # Optionally try runtime clone if WAN_REPO_URL provided
        WAN_REPO_URL = os.environ.get("WAN_REPO_URL", "")
        if WAN_REPO_URL:
            try:
                subprocess.check_call(["git", "clone", WAN_REPO_URL, WAN_REPO_DIR])
            except Exception as e:
                print("Runtime clone failed:", e)
        if not Path(GENERATE_PY).exists():
            raise FileNotFoundError(f"generate.py not found at {GENERATE_PY}. Place WAN repo in the image or set WAN_REPO_URL env var.")


def save_b64_image(b64_string, out_path="/tmp/input_from_b64.png"):
    header = ""
    if b64_string.startswith("data:"):
        header, b64_string = b64_string.split(",", 1)
    b = base64.b64decode(b64_string)
    with open(out_path, "wb") as f:
        f.write(b)
    return out_path


def run_generate(image_path, prompt, size="832*480", frame_num=48, sample_steps=25, extra_flags=None):
    """
    Calls generate.py in the WAN repo as a subprocess. Returns path to the newest produced .mp4
    """
    out_name = f"out_{uuid.uuid4().hex}.mp4"
    cmd = [
        "python", GENERATE_PY,
        "--task", "i2v-A14B",
        "--size", str(size),
        "--ckpt_dir", str(CKPT_DIR),
        "--image", str(image_path),
        "--prompt", str(prompt),
        "--frame_num", str(int(frame_num)),
        "--sample_steps", str(int(sample_steps)),
        "--offload_model", "True",
        "--convert_model_dtype",
        "--output_dir", str(OUTPUT_DIR)
    ]
    if extra_flags:
        cmd += extra_flags

    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    print("Running WAN generate command:", cmd_str)

    proc = subprocess.run(cmd, cwd=WAN_REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)

    # find latest mp4
    files = sorted(Path(OUTPUT_DIR).glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise RuntimeError("generate.py did not produce any .mp4 outputs. Check the logs above.")
    return str(files[0])


def handler(job):
    try:
        ensure_model_and_repo()
        inp = job.get("input", {})
        prompt = inp.get("prompt", "")
        image_path = inp.get("image_path", None)
        image_b64 = inp.get("image_b64", None)

        if not image_path and not image_b64:
            return {"status": "error", "message": "Provide 'image_path' (path inside container) or 'image_b64'"}

        if image_b64:
            image_path = save_b64_image(image_b64)

        size = inp.get("size", "832*480")
        frame_num = inp.get("frame_num", 48)
        sample_steps = inp.get("sample_steps", 25)

        start = time.time()
        out_video = run_generate(image_path=image_path, prompt=prompt, size=size, frame_num=frame_num, sample_steps=sample_steps)
        dur = time.time() - start
        print(f"Generated {out_video} in {dur:.1f}s")

        url = runpod.serverless.upload_file(out_video)
        return {"status": "success", "video_url": url}
    except Exception as e:
        tb = traceback.format_exc()
        print("Handler error:", tb)
        return {"status": "error", "message": str(e), "traceback": tb}


if __name__ == "__main__":
    try:
        runpod.serverless.start({"handler": handler})
    except Exception:
        print("RunPod runtime not detected. You can test handler() locally.")
