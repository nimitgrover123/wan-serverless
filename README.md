WAN 2.2 RunPod Serverless (local model via Volume)

Mount your RunPod Volume at:
    /runpod-volume/wan22

Inside volume, add:
    - config.json
    - model.safetensors

Deploy folder as Serverless → Handler = handler.py
