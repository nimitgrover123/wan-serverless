import runpod
import torch
import os
import base64
import cv2
from wan.pipeline import WANPipeline

VOLUME_PATH = "/runpod-volume/wan22"   # your Runpod Volume mount

os.makedirs(VOLUME_PATH, exist_ok=True)

print("🔄 Loading WAN 2.2 model from volume:", VOLUME_PATH)

pipe = WANPipeline.from_pretrained(
    VOLUME_PATH,
    torch_dtype=torch.float16
).to("cuda")

print("✅ WAN 2.2 Loaded!")


def decode_base64_image(b64_string):
    img_bytes = base64.b64decode(b64_string)
    numpy_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(numpy_arr, cv2.IMREAD_COLOR)


def encode_video_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_video(input_image_b64, prompt):
    img = decode_base64_image(input_image_b64)

    output_path = "/tmp/out.mp4"

    video = pipe.i2v(img, prompt=prompt)
    video.save(output_path)

    return encode_video_to_b64(output_path)


def handler(job):

    try:
        input_image = job["input"]["image"]
        prompt = job["input"].get("prompt", "high quality cinematic video")

        video_b64 = generate_video(input_image, prompt)

        return {
            "status": "success",
            "video_base64": video_b64
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


runpod.serverless.start({"handler": handler})
