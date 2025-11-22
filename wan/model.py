class WANModel:
    def __init__(self, config):
        self.config = config

    def load_state_dict(self, state):
        print("Loading weights...")

    def to(self, device):
        print("Model moved to", device)

    def generate_video(self, image, prompt):
        # WAN's real inference logic goes here
        raise NotImplementedError("Plug WAN's official inference code here.")
