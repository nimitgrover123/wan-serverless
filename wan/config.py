import json

class WANConfig:
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_json(cls, file_path):
        with open(file_path, "r") as f:
            return cls(json.load(f))
