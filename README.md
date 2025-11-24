# Deploying WAN 2.2 Serverless on RunPod (Complete)


Prereqs:
- Upload WAN model files to a RunPod Volume (recommended mount path: /runpod-volume/wan2.2-i2v)
- Ensure the WAN repo (generate.py) is included in the image or accessible via WAN_REPO_URL env var


Steps:
1. Build image locally (optional):
docker build -t ghcr.io/you/wan22-serverless:latest .
docker push ghcr.io/you/wan22-serverless:latest


2. Create RunPod Volume (name it wan2.2-i2v, size 30-80GB)
3. Upload model files into the volume at /wan2.2-i2v
4. Deploy Serverless endpoint on RunPod:
- Use the image above
- Add Volume mount: Volume=wan2.2-i2v, Mount path=/runpod-volume
- (Optional) Set env: WAN_REPO_URL, CKPT_SUBDIR, etc.
5. Test endpoint using the provided sample_request.sh


Notes:
- This image intentionally does NOT pip-install the Wan2.2 GitHub repo at build time to avoid flash-attn compile failures. The handler executes the repo's generate.py at runtime; ensure the repo is present in the image or set WAN_REPO_URL so the container can clone it at startup.
- If you want in-process imports instead of subprocess, paste the WAN generate.py or its API and I will integrate it.
