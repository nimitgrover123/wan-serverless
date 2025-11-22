import torch
from wan.model import WANModel
from wan.config import WANConfig

class WANPipeline:
    @classmethod
    def from_pretrained(cls, model_path, torch_dtype=torch.float16):
        config = WANConfig.from_json(f"{model_path}/config.json")
        model = WANModel(config)

        state = torch.load(f"{model_path}/model.safetensors", map_location="cpu")
        model.load_state_dict(state)

        model = model.to(torch_dtype)

        pipe = cls()
        pipe.model = model
        pipe.config = config
        return pipe

    def to(self, device):
        self.model.to(device)
        return self

    def i2v(self, image, prompt):
        return self.model.generate_video(image, prompt)
